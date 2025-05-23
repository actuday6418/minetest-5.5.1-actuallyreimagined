use chunk_manager::{ChunkManager, GpuChunkData};
use glam::{Mat4, f32::Vec3};
use std::{
    error::Error,
    path::Path,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};
use voxel_physics::{PhysicsBody, step_simulation};
use wgpu::util::DeviceExt;
use winit::event::ElementState;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

mod chunk_manager;
mod frustum;
mod meshing;
mod mipmap;
mod world;

use frustum::Frustum;
use glam::IVec3;
use meshing::data::{FaceData, create_quad_templates};
use world::{CHUNK_SIZE, World};

const MOUSE_SENSITIVITY: f32 = 0.01;
const MOVE_SPEED: f32 = 0.5;
const CHUNK_RADIUS: i32 = 15;
const CHUNK_RADIUS_VERTICAL: i32 = 2;
const MAX_UPLOADS_PER_FRAME: usize = 20;
const MSAA_SAMPLES: u32 = 4;
const MAX_MIP_LEVELS: u32 = 11;
const PLAYER_HEIGHT: f32 = 1.8;
const PLAYER_WIDTH: f32 = 0.6;
const PLAYER_EYE_LEVEL: f32 = 1.6; // Relative to feet
const PLAYER_HALF_WIDTH: f32 = PLAYER_WIDTH / 2.0;
const PLAYER_BODY_DIMS: Vec3 = Vec3::new(PLAYER_WIDTH, PLAYER_HEIGHT, PLAYER_WIDTH);
const GRAVITY: f32 = 25.0;
const MAX_SPEED: f32 = 6.0;
const JUMP_STRENGTH: f32 = 8.0;

#[derive(Debug, thiserror::Error)]
enum AppError {
    #[error("WGPU Request Device Error: {0}")]
    WgpuRequestDeviceError(#[from] wgpu::RequestDeviceError),
    #[error("IO Error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Image Error: {0}")]
    ImageError(#[from] image::ImageError),
    #[error("WGPU Surface Error: {0}")]
    SurfaceError(#[from] wgpu::SurfaceError),
    #[error("Application Error: {0}")]
    Custom(String),
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let event_loop = EventLoop::new()?;
    let mut app = pollster::block_on(App::new())?;
    event_loop.run_app(&mut app)?;
    Ok(())
}

struct WgpuState {
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    config: wgpu::SurfaceConfiguration,
    fill_render_pipeline: wgpu::RenderPipeline,
    line_render_pipeline: wgpu::RenderPipeline,
    texture_bind_group: wgpu::BindGroup,
    uniform_bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    shared_index_buffer: wgpu::Buffer,
    depth_texture_view: wgpu::TextureView,
    msaa_texture_view: Option<wgpu::TextureView>,
    window: Arc<Window>,
    size: PhysicalSize<u32>,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    sun_direction: [f32; 3],
    _padding: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ChunkPushConstants {
    chunk_offset_world: [f32; 3],
    _padding: u32,
}

struct App {
    wgpu_state: Option<WgpuState>,
    chunk_manager: ChunkManager,
    fps_last_instant: Instant,
    last_update_instant: Instant,
    fps_frame_count: u32,
    player_body: PhysicsBody,
    player_yaw: f32,
    player_pitch: f32,
    move_direction: Vec3,
    jump_requested: bool,
    current_frustum: Frustum,
    wiremesh_mode: bool,
    is_focused: bool,
}

impl App {
    async fn new() -> Result<Self, AppError> {
        let initial_cam_pos = Vec3::new(0.0, 525.0, 0.0);
        let initial_yaw: f32 = std::f32::consts::FRAC_PI_2;
        let initial_pitch: f32 = 0.0;
        let player_start_pos =
            initial_cam_pos - Vec3::Y * PLAYER_EYE_LEVEL + Vec3::Y * PLAYER_HEIGHT * 0.5;
        let player_body = PhysicsBody::new(player_start_pos, PLAYER_BODY_DIMS);

        let world = Arc::new(RwLock::new(World::new()));
        let current_frustum = Frustum::from_view_proj(&Mat4::ZERO);
        let chunk_manager = ChunkManager::new(
            CHUNK_RADIUS,
            CHUNK_RADIUS_VERTICAL,
            MAX_UPLOADS_PER_FRAME,
            world.clone(),
        );

        let app = App {
            wgpu_state: None,
            chunk_manager,
            fps_last_instant: Instant::now(),
            last_update_instant: Instant::now(),
            fps_frame_count: 0,
            player_body,
            player_yaw: initial_yaw,
            player_pitch: initial_pitch,
            is_focused: false,
            move_direction: Vec3::ZERO,
            current_frustum,
            wiremesh_mode: false,
            jump_requested: false,
        };

        Ok(app)
    }

    async fn init_wgpu(&mut self, window: Arc<Window>) -> Result<(), AppError> {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        log::info!("Using adapter: {:?}", adapter.get_info());

        let required_features = wgpu::Features::PUSH_CONSTANTS | wgpu::Features::POLYGON_MODE_LINE;
        let required_limits = wgpu::Limits {
            max_push_constant_size: std::mem::size_of::<ChunkPushConstants>() as u32,
            ..wgpu::Limits::default()
        }
        .using_resolution(adapter.limits());

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features,
                required_limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let diffuse_image = image::open(Path::new("assets/atlas_padded.png"))?.to_rgba8();
        let diffuse_dimensions = diffuse_image.dimensions();
        let diffuse_texture_size = wgpu::Extent3d {
            width: diffuse_dimensions.0,
            height: diffuse_dimensions.1,
            depth_or_array_layers: 1,
        };

        let diffuse_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: diffuse_texture_size,
            mip_level_count: MAX_MIP_LEVELS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: Some("Diffuse Texture"),
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &diffuse_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &diffuse_image,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * diffuse_dimensions.0),
                rows_per_image: Some(diffuse_dimensions.1),
            },
            diffuse_texture_size,
        );

        let diffuse_texture_view =
            diffuse_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let diffuse_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Diffuse Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            anisotropy_clamp: 1,
            lod_min_clamp: 0.0,
            lod_max_clamp: (MAX_MIP_LEVELS - 1) as f32,
            ..Default::default()
        });

        let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Mipmap Encoder"),
        });
        mipmap::generate(
            &device,
            &mut command_encoder,
            &diffuse_texture,
            MAX_MIP_LEVELS,
        );
        queue.submit(std::iter::once(command_encoder.finish()));

        let quad_template_data = create_quad_templates();
        let shared_quad_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Quad Template Buffer"),
            contents: bytemuck::cast_slice(&quad_template_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let indices: [u32; 6] = [0, 1, 2, 3, 4, 5];
        let shared_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Shared Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Texture Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Uniform Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    // Uniform Buffer
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Texture Bind Group"),
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: shared_quad_buffer.as_entire_binding(),
                },
            ],
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/shader.wgsl"));

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&uniform_bind_group_layout, &texture_bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::VERTEX,
                range: 0..std::mem::size_of::<ChunkPushConstants>() as u32,
            }],
        });

        let color_target_state = [Some(wgpu::ColorTargetState {
            format: config.format,
            blend: Some(wgpu::BlendState::REPLACE),
            write_mask: wgpu::ColorWrites::ALL,
        })];

        let base_pipeline_desc = wgpu::RenderPipelineDescriptor {
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[FaceData::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &color_target_state,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                conservative: false,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: MSAA_SAMPLES,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
            label: None,
        };

        let mut fill_pipeline_desc = base_pipeline_desc.clone();
        fill_pipeline_desc.label = Some("Fill Render Pipeline");
        fill_pipeline_desc.primitive.polygon_mode = wgpu::PolygonMode::Fill;
        let fill_render_pipeline = device.create_render_pipeline(&fill_pipeline_desc);

        let mut line_pipeline_desc = base_pipeline_desc;
        line_pipeline_desc.label = Some("Line Render Pipeline");
        line_pipeline_desc.primitive.polygon_mode = wgpu::PolygonMode::Line;
        let line_render_pipeline = device.create_render_pipeline(&line_pipeline_desc);

        let (depth_texture_view, msaa_texture_view) =
            Self::create_size_dependent_textures(&device, &config, MSAA_SAMPLES);

        self.wgpu_state = Some(WgpuState {
            surface,
            device,
            queue,
            config,
            fill_render_pipeline,
            line_render_pipeline,
            texture_bind_group,
            uniform_bind_group,
            uniform_buffer,
            shared_index_buffer,
            depth_texture_view,
            msaa_texture_view,
            window,
            size,
        });

        Ok(())
    }

    fn create_size_dependent_textures(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        sample_count: u32,
    ) -> (wgpu::TextureView, Option<wgpu::TextureView>) {
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_texture_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let msaa_texture_view = if sample_count > 1 {
            let msaa_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("MSAA Texture"),
                size: wgpu::Extent3d {
                    width: config.width,
                    height: config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: wgpu::TextureDimension::D2,
                format: config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            Some(msaa_texture.create_view(&wgpu::TextureViewDescriptor::default()))
        } else {
            None
        };

        (depth_texture_view, msaa_texture_view)
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if let Some(state) = self.wgpu_state.as_mut() {
            if new_size.width > 0 && new_size.height > 0 {
                state.size = new_size;
                state.config.width = new_size.width;
                state.config.height = new_size.height;
                state.surface.configure(&state.device, &state.config);
                let (depth_view, msaa_view) = Self::create_size_dependent_textures(
                    &state.device,
                    &state.config,
                    MSAA_SAMPLES,
                );
                state.depth_texture_view = depth_view;
                state.msaa_texture_view = msaa_view;
            }
            state.window.request_redraw();
        }
    }

    fn render(&mut self) -> Result<(), AppError> {
        let camera_position = self.get_camera_position();
        let state = self
            .wgpu_state
            .as_mut()
            .ok_or_else(|| AppError::Custom("WGPU State not initialized".to_string()))?;

        let output_surface_texture = match state.surface.get_current_texture() {
            Ok(tex) => tex,
            Err(wgpu::SurfaceError::Lost) => {
                log::warn!("Surface lost, recreating");
                let size = state.size;
                self.resize(size);
                return Ok(());
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                return Err(AppError::SurfaceError(wgpu::SurfaceError::OutOfMemory));
            }
            Err(wgpu::SurfaceError::Outdated) => {
                log::warn!("Surface outdated");
                return Ok(());
            }
            Err(e) => return Err(AppError::SurfaceError(e)),
        };

        let output_view = output_surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let aspect_ratio = state.config.width as f32 / state.config.height as f32;
        let forward = Vec3::new(
            self.player_yaw.cos() * self.player_pitch.cos(),
            self.player_pitch.sin(),
            self.player_yaw.sin() * self.player_pitch.cos(),
        )
        .normalize();

        let view = Mat4::look_at_rh(camera_position, camera_position + forward, Vec3::Y);
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, aspect_ratio, 0.1, 2000.0);
        let view_proj = proj * view;

        self.current_frustum = Frustum::from_view_proj(&view_proj);

        let sun = Vec3::new(0.48, 0.98, 0.78).normalize();
        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            sun_direction: sun.to_array(),
            _padding: 0,
        };
        state
            .queue
            .write_buffer(&state.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        let mut visible_chunks: Vec<(IVec3, &GpuChunkData, f32)> = self
            .chunk_manager
            .get_renderable_chunks(&self.current_frustum, camera_position)
            .collect();
        visible_chunks
            .sort_unstable_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        {
            let color_attachment = if let Some(msaa_view) = &state.msaa_texture_view {
                wgpu::RenderPassColorAttachment {
                    view: msaa_view,
                    resolve_target: Some(&output_view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.6,
                            g: 0.8,
                            b: 1.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }
            } else {
                wgpu::RenderPassColorAttachment {
                    view: &output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.6,
                            g: 0.8,
                            b: 1.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }
            };

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(color_attachment)],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &state.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            if self.wiremesh_mode {
                render_pass.set_pipeline(&state.line_render_pipeline);
            } else {
                render_pass.set_pipeline(&state.fill_render_pipeline);
            }
            render_pass.set_bind_group(0, &state.uniform_bind_group, &[]);
            render_pass.set_bind_group(1, &state.texture_bind_group, &[]);
            render_pass.set_index_buffer(
                state.shared_index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );

            for (chunk_coords, data, _dist_sq) in visible_chunks {
                if let Some(face_buffer) = &data.face_buffer {
                    if data.face_count > 0 {
                        let chunk_coords_global =
                            chunk_coords.map(|e| e * CHUNK_SIZE as i32).as_vec3();

                        let push_constants = ChunkPushConstants {
                            chunk_offset_world: chunk_coords_global.to_array(),
                            _padding: 0,
                        };

                        render_pass.set_push_constants(
                            wgpu::ShaderStages::VERTEX,
                            0,
                            bytemuck::cast_slice(&[push_constants]),
                        );

                        render_pass.set_vertex_buffer(0, face_buffer.slice(..));
                        render_pass.draw_indexed(0..6, 0, 0..data.face_count);
                    }
                }
            }
        }

        state.queue.submit(std::iter::once(encoder.finish()));
        output_surface_texture.present();
        Ok(())
    }

    fn update_fps_counter(&mut self) {
        self.fps_frame_count += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(self.fps_last_instant);
        if elapsed >= Duration::from_secs(1) {
            let fps = self.fps_frame_count as f64 / elapsed.as_secs_f64();
            log::info!("FPS: {:>6.2}", fps,);
            self.fps_frame_count = 0;
            self.fps_last_instant = now;
        }
    }

    fn get_camera_position(&self) -> Vec3 {
        self.player_body.position + Vec3::Y * (PLAYER_EYE_LEVEL - PLAYER_HEIGHT * 0.5)
    }

    fn update(&mut self, dt: f32) {
        self.player_body.velocity.y -= GRAVITY * dt;

        let forward =
            Vec3::new(self.player_yaw.cos(), 0.0, self.player_yaw.sin()).normalize_or_zero();
        let right = Vec3::new(-forward.z, 0.0, forward.x);

        let mut desired_velocity = Vec3::ZERO;
        desired_velocity += forward * self.move_direction.z;
        desired_velocity += right * self.move_direction.x;

        if desired_velocity.length_squared() > 0.0 {
            desired_velocity = desired_velocity.normalize() * MAX_SPEED;
        }
        self.player_body.velocity.x = desired_velocity.x;
        self.player_body.velocity.z = desired_velocity.z;

        if self.jump_requested && self.player_body.is_grounded {
            self.player_body.velocity.y = JUMP_STRENGTH;
        }
        self.jump_requested = false;

        {
            let world_reader = self.chunk_manager.world.read().unwrap();
            step_simulation(&mut self.player_body, dt, &*world_reader);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.wgpu_state.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(Window::default_attributes().with_title("WGPU Voxel Renderer"))
                    .expect("Failed to create window"),
            );

            match pollster::block_on(self.init_wgpu(window)) {
                Ok(_) => log::info!("WGPU Initialized Successfully"),
                Err(e) => {
                    log::error!("Failed to initialize WGPU: {}", e);
                    event_loop.exit();
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let state = match self.wgpu_state.as_mut() {
            Some(s) => s,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => {
                log::info!("Close Requested");
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                log::info!("Window Resized: {:?}", physical_size);
                self.resize(physical_size);
            }

            WindowEvent::ScaleFactorChanged { .. } => {
                let new_inner_size = state.window.inner_size();
                log::info!("Scale Factor Changed, New Size: {:?}", new_inner_size);
                self.resize(new_inner_size);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let is_pressed = event.state == ElementState::Pressed;
                if self.is_focused {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::KeyW) => {
                            self.move_direction.z = if is_pressed { 1.0 } else { 0.0 }
                        }
                        PhysicalKey::Code(KeyCode::KeyS) => {
                            self.move_direction.z = if is_pressed { -1.0 } else { 0.0 }
                        }
                        PhysicalKey::Code(KeyCode::KeyA) => {
                            self.move_direction.x = if is_pressed { -1.0 } else { 0.0 }
                        }
                        PhysicalKey::Code(KeyCode::KeyD) => {
                            self.move_direction.x = if is_pressed { 1.0 } else { 0.0 }
                        }
                        PhysicalKey::Code(KeyCode::Space) => {
                            if is_pressed {
                                self.jump_requested = true;
                            }
                        }
                        PhysicalKey::Code(KeyCode::ShiftLeft) => {
                            self.move_direction.y = if is_pressed { -1.0 } else { 0.0 }
                        }
                        PhysicalKey::Code(KeyCode::KeyM) => {
                            if is_pressed {
                                self.wiremesh_mode = !self.wiremesh_mode;
                            }
                        }
                        _ => {}
                    }
                }
                if event.physical_key == PhysicalKey::Code(KeyCode::Escape) && is_pressed {
                    self.is_focused = !self.is_focused;
                    if self.is_focused {
                        let _ = state
                            .window
                            .set_cursor_grab(winit::window::CursorGrabMode::Confined);
                        state.window.set_cursor_visible(false);
                    } else {
                        let _ = state
                            .window
                            .set_cursor_grab(winit::window::CursorGrabMode::None);
                        state.window.set_cursor_visible(true);
                    }
                }
            }
            WindowEvent::MouseInput {
                state: button_state,
                button,
                ..
            } => {
                if !self.is_focused
                    && button_state == ElementState::Pressed
                    && button == winit::event::MouseButton::Left
                {
                    self.is_focused = true;
                    let _ = state
                        .window
                        .set_cursor_grab(winit::window::CursorGrabMode::Confined);
                    state.window.set_cursor_visible(false);
                }
            }
            WindowEvent::RedrawRequested => {
                let size = state.size;
                self.update_fps_counter();

                match self.render() {
                    Ok(_) => {}
                    Err(AppError::SurfaceError(wgpu::SurfaceError::Lost)) => self.resize(size),
                    Err(AppError::SurfaceError(wgpu::SurfaceError::Outdated)) => {
                        log::warn!("Surface outdated during render - likely resize race")
                    }
                    Err(AppError::SurfaceError(wgpu::SurfaceError::OutOfMemory)) => {
                        log::error!("Out of memory!");
                        event_loop.exit();
                    }
                    Err(e) => log::error!("Error during render: {:?}", e),
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        match event {
            DeviceEvent::MouseMotion { delta: (dx, dy) } => {
                if self.is_focused {
                    self.player_yaw += (dx as f32) * MOUSE_SENSITIVITY;
                    self.player_pitch -= (dy as f32) * MOUSE_SENSITIVITY;
                    self.player_pitch = self.player_pitch.clamp(
                        -std::f32::consts::FRAC_PI_2 + 0.01,
                        std::f32::consts::FRAC_PI_2 - 0.01,
                    );
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_update_instant).as_secs_f32();
        self.last_update_instant = now;

        let dt = dt.min(1.0 / 20.0);

        if self.is_focused {
            self.update(dt);
        }
        if let Some(state) = &self.wgpu_state {
            self.chunk_manager
                .update(self.get_camera_position(), state.device.clone());
            state.window.request_redraw();
        }
    }
}
