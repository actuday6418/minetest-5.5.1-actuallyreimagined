use crossbeam_channel::{Receiver, Sender, unbounded};
use glam::{Mat4, f32::Vec3};
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::{
    collections::{HashMap, HashSet},
    error::Error,
    path::Path,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};
use vulkano::{
    Validated, VulkanError, VulkanLibrary,
    buffer::{
        Buffer, BufferCreateInfo, BufferUsage, Subbuffer,
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
    },
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo,
        allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceOwned, Queue, QueueCreateInfo,
        QueueFlags, physical::PhysicalDeviceType,
    },
    format::Format,
    image::{
        Image, ImageCreateInfo, ImageType, ImageUsage,
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::{DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::{PolygonMode, RasterizationState},
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::EntryPoint,
    swapchain::{
        Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo, acquire_next_image,
    },
    sync::{self, GpuFuture},
};
use winit::event::ElementState;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

mod world;
mod worldgen;

use world::{ChunkBlocks, ChunkCoords, World};
use worldgen::{ATLAS_H, ATLAS_W, CHUNK_BREADTH, CHUNK_HEIGHT, VertexData, generate_chunk_mesh};

const MOUSE_SENSITIVITY: f32 = 0.01;
const MOVE_SPEED: f32 = 0.5;
const CHUNK_RADIUS: i32 = 30;
const MAX_UPLOADS_PER_FRAME: usize = 20;
const MAX_QUADS_PER_CHUNK: usize = CHUNK_BREADTH * CHUNK_BREADTH * CHUNK_HEIGHT * 6;
const MAX_INDICES_PER_CHUNK: usize = MAX_QUADS_PER_CHUNK * 6;

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}

struct ChunkData {
    vertex_buffer: Subbuffer<[VertexData]>,
}

struct CpuMeshData {
    vertices: Vec<VertexData>,
}

type MeshGenResult = (ChunkCoords, CpuMeshData);

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    shared_index_buffer: Subbuffer<[u32]>,
    world: Arc<RwLock<World>>,
    loaded_chunks: HashMap<ChunkCoords, ChunkData>,
    generating_chunks: HashSet<ChunkCoords>,
    pending_upload: HashMap<ChunkCoords, CpuMeshData>,
    texture_view: Arc<ImageView>,
    texture_sampler: Arc<Sampler>,
    uniform_buffer_allocator: SubbufferAllocator,
    rcx: Option<RenderContext>,
    fps_last_instant: Instant,
    fps_frame_count: u32,
    camera_position: Vec3,
    last_camera_chunk_coords: ChunkCoords,
    yaw: f32,
    pitch: f32,
    is_focused: bool,
    is_moving_forward: bool,
    is_moving_backward: bool,
    is_moving_left: bool,
    is_moving_right: bool,
    is_moving_up: bool,
    is_moving_down: bool,
    thread_pool: Arc<ThreadPool>,
    result_sender: Sender<MeshGenResult>,
    result_receiver: Receiver<MeshGenResult>,
    all_uploads_next_frame: bool, // set this to trigger all uploads next frame, regardless of MAX_UPLOADS_PER_FRAME
}

struct RenderContext {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    vs: EntryPoint,
    fs: EntryPoint,
    pipeline: Arc<GraphicsPipeline>,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Self {
        let library = VulkanLibrary::new().unwrap();
        let required_extensions = Surface::required_extensions(event_loop).unwrap();
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.presentation_support(i as u32, event_loop).unwrap()
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .unwrap();

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let image = image::open(Path::new("assets/atlas.png"))
            .expect("Failed to open texture file")
            .to_rgba8();
        let image_data = image.into_raw();
        let texture_image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_UNORM,
                extent: [ATLAS_W as u32, ATLAS_H as u32, 1],
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();
        let texture_view = ImageView::new_default(texture_image.clone()).unwrap();
        let texture_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                mipmap_mode: vulkano::image::sampler::SamplerMipmapMode::Nearest,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..Default::default()
            },
        )
        .unwrap();
        let upload_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            image_data,
        )
        .unwrap();
        let mut upload_cb_builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        upload_cb_builder
            .copy_buffer_to_image(
                vulkano::command_buffer::CopyBufferToImageInfo::buffer_image(
                    upload_buffer,
                    texture_image.clone(),
                ),
            )
            .unwrap();
        let upload_cb = upload_cb_builder.build().unwrap();
        let future = sync::now(device.clone())
            .then_execute(queue.clone(), upload_cb)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();

        let uniform_buffer_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let initial_camera_position = Vec3::new(0.0, 20.0, 0.0);
        let initial_chunk_coords = Self::get_chunk_coords_at(initial_camera_position);

        let thread_pool = Arc::new(ThreadPoolBuilder::new().num_threads(12).build().unwrap());
        let (result_sender, result_receiver) = unbounded::<MeshGenResult>();

        let world = Arc::new(RwLock::new(World::new()));
        let mut indices = Vec::with_capacity(MAX_INDICES_PER_CHUNK);
        for i in 0..(MAX_QUADS_PER_CHUNK as u32) {
            let base_vertex = i * 4;
            indices.extend_from_slice(&[
                base_vertex,
                base_vertex + 1,
                base_vertex + 2,
                base_vertex + 2,
                base_vertex + 3,
                base_vertex,
            ]);
        }

        let staging_index_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            indices,
        )
        .unwrap();

        let shared_index_buffer = Buffer::new_slice(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            MAX_INDICES_PER_CHUNK as u64,
        )
        .unwrap();

        let mut upload_cb_builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        upload_cb_builder
            .copy_buffer(vulkano::command_buffer::CopyBufferInfo::buffers(
                staging_index_buffer.clone(),
                shared_index_buffer.clone(),
            ))
            .unwrap();

        let upload_cb = upload_cb_builder.build().unwrap();

        let future = sync::now(device.clone())
            .then_execute(queue.clone(), upload_cb)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();

        let mut app = App {
            instance,
            device,
            queue,
            fps_last_instant: Instant::now(),
            fps_frame_count: 0,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
            shared_index_buffer,
            world,
            loaded_chunks: HashMap::new(),
            generating_chunks: HashSet::new(),
            pending_upload: HashMap::new(),
            texture_view,
            texture_sampler,
            uniform_buffer_allocator,
            rcx: None,
            camera_position: initial_camera_position,
            last_camera_chunk_coords: initial_chunk_coords,
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: -0.0,
            is_focused: false,
            is_moving_forward: false,
            is_moving_backward: false,
            is_moving_left: false,
            is_moving_right: false,
            is_moving_up: false,
            is_moving_down: false,
            all_uploads_next_frame: true,
            thread_pool,
            result_sender,
            result_receiver,
        };

        app.request_chunks_around_camera();
        app
    }

    fn get_chunk_coords_at(position: Vec3) -> ChunkCoords {
        (
            (position.x / CHUNK_BREADTH as f32).floor() as i32,
            (position.z / CHUNK_BREADTH as f32).floor() as i32,
        )
    }

    fn request_chunks_around_camera(&mut self) {
        const DIRECT_NEIGHBOR_OFFSETS: [(i32, i32); 4] = [(0, 1), (0, -1), (1, 0), (-1, 0)];
        let camera_coords = Self::get_chunk_coords_at(self.camera_position);

        let mut required_coords = HashSet::new();
        for x in (camera_coords.0 - CHUNK_RADIUS)..=(camera_coords.0 + CHUNK_RADIUS) {
            for z in (camera_coords.1 - CHUNK_RADIUS)..=(camera_coords.1 + CHUNK_RADIUS) {
                required_coords.insert((x, z));
            }
        }

        let current_mesh_coords: HashSet<ChunkCoords> =
            self.loaded_chunks.keys().cloned().collect();
        let current_pending_coords: HashSet<ChunkCoords> =
            self.pending_upload.keys().cloned().collect();
        let all_known_coords = current_mesh_coords
            .union(&current_pending_coords)
            .cloned()
            .collect::<HashSet<_>>()
            .union(&self.generating_chunks)
            .cloned()
            .collect::<HashSet<_>>();

        let mut neighbors_to_remesh = HashSet::new();

        let coords_to_unload = all_known_coords
            .difference(&required_coords)
            .cloned()
            .collect::<Vec<_>>();

        for &coord in &coords_to_unload {
            self.loaded_chunks.remove(&coord);
            self.pending_upload.remove(&coord);
            self.generating_chunks.remove(&coord);

            for (dx, dz) in DIRECT_NEIGHBOR_OFFSETS {
                let neighbor_coord = (coord.0 + dx, coord.1 + dz);
                if required_coords.contains(&neighbor_coord)
                    && self.loaded_chunks.contains_key(&neighbor_coord)
                {
                    neighbors_to_remesh.insert(neighbor_coord);
                }
            }
        }

        let mut coords_to_generate = required_coords
            .difference(&all_known_coords)
            .cloned()
            .collect::<HashSet<_>>();

        for coord_to_remesh in neighbors_to_remesh {
            self.loaded_chunks.remove(&coord_to_remesh);
            self.pending_upload.remove(&coord_to_remesh);
            self.generating_chunks.remove(&coord_to_remesh);
            coords_to_generate.insert(coord_to_remesh);
        }

        for coord in coords_to_generate {
            if self.generating_chunks.contains(&coord)
                || self.pending_upload.contains_key(&coord)
                || self.loaded_chunks.contains_key(&coord)
            {
                continue;
            }
            self.generating_chunks.insert(coord);

            let world_arc = self.world.clone();
            let pool = self.thread_pool.clone();
            let result_sender = self.result_sender.clone();

            pool.spawn(move || {
                let chunk_blocks = ChunkBlocks::generate(coord);

                {
                    let mut world_writer = world_arc.write().unwrap();
                    world_writer.insert_chunk_blocks(coord, Arc::new(chunk_blocks));

                    for (dx, dz) in DIRECT_NEIGHBOR_OFFSETS {
                        let neighbor_coord = (coord.0 + dx, coord.1 + dz);
                        world_writer.ensure_chunk_generated(neighbor_coord);
                    }
                }

                let vertices = {
                    let world_reader = world_arc.read().unwrap();
                    generate_chunk_mesh(coord, &world_reader)
                };

                let cpu_mesh_data = CpuMeshData { vertices };

                let _ = result_sender.send((coord, cpu_mesh_data));
            });
        }
        self.last_camera_chunk_coords = camera_coords;
    }

    fn process_pending_uploads(&mut self) {
        while let Ok((coords, cpu_mesh_data)) = self.result_receiver.try_recv() {
            if self.generating_chunks.remove(&coords) {
                if !self.loaded_chunks.contains_key(&coords) {
                    self.pending_upload.insert(coords, cpu_mesh_data);
                }
            }
        }

        let mut uploaded_count = 0;
        let mut coords_uploaded = Vec::new();

        for (coord, cpu_data) in self.pending_upload.iter() {
            if uploaded_count >= MAX_UPLOADS_PER_FRAME && !self.all_uploads_next_frame {
                break;
            }

            let camera_coords = Self::get_chunk_coords_at(self.camera_position);
            let dx = (coord.0 - camera_coords.0).abs();
            let dz = (coord.1 - camera_coords.1).abs();
            if dx > CHUNK_RADIUS || dz > CHUNK_RADIUS {
                coords_uploaded.push(*coord);
                continue;
            }

            let vertex_buffer = Buffer::from_iter(
                self.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                cpu_data.vertices.clone(),
            )
            .unwrap();

            let chunk_data = ChunkData { vertex_buffer };

            self.loaded_chunks.insert(*coord, chunk_data);
            coords_uploaded.push(*coord);
            uploaded_count += 1;
        }

        for coord in coords_uploaded {
            self.pending_upload.remove(&coord);
        }
        if self.all_uploads_next_frame {
            self.all_uploads_next_frame = false;
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        let surface = Surface::from_window(self.instance.clone(), window.clone()).unwrap();
        let window_size = window.inner_size();

        let (swapchain, images) = {
            let surface_capabilities = self
                .device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();
            let (image_format, _) = self
                .device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0];

            Swapchain::new(
                self.device.clone(),
                surface,
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),
                    image_format,
                    image_extent: window_size.into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let render_pass = vulkano::single_pass_renderpass!(
            self.device.clone(),
            attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
                depth_stencil: {
                    format: Format::D16_UNORM,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {depth_stencil},
            },
        )
        .unwrap();

        let vs = vs::load(self.device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = fs::load(self.device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let (framebuffers, pipeline) = window_size_dependent_setup(
            window_size,
            &images,
            &render_pass,
            &self.memory_allocator,
            &vs,
            &fs,
        );

        let previous_frame_end = Some(sync::now(self.device.clone()).boxed());

        self.rcx = Some(RenderContext {
            window,
            swapchain,
            render_pass,
            framebuffers,
            vs,
            fs,
            pipeline,
            recreate_swapchain: false,
            previous_frame_end,
        });
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
                    self.yaw += (dx as f32) * MOUSE_SENSITIVITY;
                    self.pitch -= (dy as f32) * MOUSE_SENSITIVITY;
                    self.pitch = self.pitch.clamp(
                        -std::f32::consts::FRAC_PI_2 + 0.01,
                        std::f32::consts::FRAC_PI_2 - 0.01,
                    );
                }
            }
            _ => {}
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let rcx = match self.rcx.as_mut() {
            Some(rcx) => rcx,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let is_pressed = event.state == ElementState::Pressed;
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::KeyW) => self.is_moving_forward = is_pressed,
                    PhysicalKey::Code(KeyCode::KeyS) => self.is_moving_backward = is_pressed,
                    PhysicalKey::Code(KeyCode::KeyA) => self.is_moving_left = is_pressed,
                    PhysicalKey::Code(KeyCode::KeyD) => self.is_moving_right = is_pressed,
                    PhysicalKey::Code(KeyCode::Space) => self.is_moving_up = is_pressed,
                    PhysicalKey::Code(KeyCode::ShiftLeft) => self.is_moving_down = is_pressed,
                    PhysicalKey::Code(KeyCode::Escape) => {
                        if is_pressed {
                            self.is_focused = false;
                            rcx.window
                                .set_cursor_grab(winit::window::CursorGrabMode::None)
                                .unwrap();
                            rcx.window.set_cursor_visible(true);
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if !self.is_focused
                    && state == ElementState::Pressed
                    && button == winit::event::MouseButton::Left
                {
                    self.is_focused = true;
                    rcx.window
                        .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                        .unwrap();
                    rcx.window.set_cursor_visible(false);
                }
            }
            WindowEvent::Focused(focused) => {
                self.is_focused = focused;
                if !focused {
                    rcx.window
                        .set_cursor_grab(winit::window::CursorGrabMode::None)
                        .unwrap();
                    rcx.window.set_cursor_visible(true);
                } else {
                    rcx.window
                        .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                        .unwrap();
                    rcx.window.set_cursor_visible(false);
                }
            }
            WindowEvent::Resized(_) => {
                rcx.recreate_swapchain = true;
            }
            WindowEvent::RedrawRequested => {
                let window_size = rcx.window.inner_size();

                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                rcx.previous_frame_end.as_mut().unwrap().cleanup_finished();

                self.fps_frame_count += 1;
                let now = Instant::now();
                let elapsed = now.duration_since(self.fps_last_instant);
                if elapsed >= Duration::from_secs(1) {
                    let fps = self.fps_frame_count as f64 / elapsed.as_secs_f64();
                    println!(
                        "FPS: {:.2} | Loaded Chunks: {} | Pending Upload: {} | Generating: {}",
                        fps,
                        self.loaded_chunks.len(),
                        self.pending_upload.len(),
                        self.generating_chunks.len()
                    );
                    self.fps_frame_count = 0;
                    self.fps_last_instant = now;
                }

                if rcx.recreate_swapchain {
                    let (new_swapchain, new_images) =
                        match rcx.swapchain.recreate(SwapchainCreateInfo {
                            image_extent: window_size.into(),
                            ..rcx.swapchain.create_info()
                        }) {
                            Ok(r) => r,
                            Err(e) => {
                                println!("Failed to recreate swapchain: {}", e);
                                return;
                            }
                        };

                    rcx.swapchain = new_swapchain;
                    (rcx.framebuffers, rcx.pipeline) = window_size_dependent_setup(
                        window_size,
                        &new_images,
                        &rcx.render_pass,
                        &self.memory_allocator,
                        &rcx.vs,
                        &rcx.fs,
                    );
                    rcx.recreate_swapchain = false;
                }

                let uniform_buffer = {
                    let aspect_ratio = rcx.swapchain.image_extent()[0] as f32
                        / rcx.swapchain.image_extent()[1] as f32;

                    let proj = Mat4::perspective_rh(
                        std::f32::consts::FRAC_PI_2,
                        aspect_ratio,
                        0.1,
                        2000.0,
                    );

                    let forward = Vec3::new(
                        self.yaw.cos() * self.pitch.cos(),
                        self.pitch.sin(),
                        self.yaw.sin() * self.pitch.cos(),
                    )
                    .normalize();

                    let vk_fix = Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0));
                    let view = Mat4::look_at_rh(
                        self.camera_position,
                        self.camera_position + forward,
                        Vec3::Y,
                    );
                    let sun = Vec3::new(0.48, 0.98, 0.78).normalize();

                    let uniform_data = vs::Data {
                        world: Mat4::IDENTITY.to_cols_array_2d(),
                        view: view.to_cols_array_2d(),
                        proj: (vk_fix * proj).to_cols_array_2d(),
                        sun: sun.to_array(),
                    };

                    let buffer = self.uniform_buffer_allocator.allocate_sized().unwrap();
                    *buffer.write().unwrap() = uniform_data;
                    buffer
                };

                let layout = &rcx.pipeline.layout().set_layouts()[0];
                let descriptor_set = DescriptorSet::new(
                    self.descriptor_set_allocator.clone(),
                    layout.clone(),
                    [
                        WriteDescriptorSet::buffer(0, uniform_buffer),
                        WriteDescriptorSet::image_view_sampler(
                            1,
                            self.texture_view.clone(),
                            self.texture_sampler.clone(),
                        ),
                    ],
                    [],
                )
                .unwrap();

                let (image_index, suboptimal, acquire_future) = match acquire_next_image(
                    rcx.swapchain.clone(),
                    None,
                )
                .map_err(Validated::unwrap)
                {
                    Ok(r) => r,
                    Err(VulkanError::OutOfDate) => {
                        rcx.recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("failed to acquire next image: {e}"),
                };

                if suboptimal {
                    rcx.recreate_swapchain = true;
                }

                let mut builder = AutoCommandBufferBuilder::primary(
                    self.command_buffer_allocator.clone(),
                    self.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![
                                Some([0.6, 0.8, 1.0, 1.0].into()),
                                Some(1f32.into()),
                            ],
                            ..RenderPassBeginInfo::framebuffer(
                                rcx.framebuffers[image_index as usize].clone(),
                            )
                        },
                        Default::default(),
                    )
                    .unwrap()
                    .bind_pipeline_graphics(rcx.pipeline.clone())
                    .unwrap()
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        rcx.pipeline.layout().clone(),
                        0,
                        descriptor_set,
                    )
                    .unwrap()
                    .bind_index_buffer(self.shared_index_buffer.clone())
                    .unwrap();

                for chunk_data in self.loaded_chunks.values() {
                    if chunk_data.vertex_buffer.len() > 0 {
                        let index_count = chunk_data.vertex_buffer.len() as u32 / 24 * 36;
                        builder
                            .bind_vertex_buffers(0, (chunk_data.vertex_buffer.clone(),))
                            .unwrap();
                        unsafe { builder.draw_indexed(index_count, 1, 0, 0, 0) }.unwrap();
                    }
                }

                builder.end_render_pass(Default::default()).unwrap();

                let command_buffer = builder.build().unwrap();
                let future = rcx
                    .previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(self.queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        self.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            rcx.swapchain.clone(),
                            image_index,
                        ),
                    )
                    .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        rcx.previous_frame_end = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        rcx.recreate_swapchain = true;
                        rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("failed to flush future: {e}");
                        rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let forward = Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize_or_zero();
        let right = -Vec3::Y.cross(forward).normalize();
        let mut move_delta = Vec3::ZERO;
        if self.is_moving_forward {
            move_delta += forward;
        }
        if self.is_moving_backward {
            move_delta -= forward;
        }
        if self.is_moving_right {
            move_delta += right;
        }
        if self.is_moving_left {
            move_delta -= right;
        }
        if self.is_moving_up {
            move_delta += Vec3::Y;
        }
        if self.is_moving_down {
            move_delta -= Vec3::Y;
        }

        let mut camera_moved_chunks = false;
        if move_delta != Vec3::ZERO {
            self.camera_position += move_delta.normalize() * MOVE_SPEED;
            let current_chunk_coords = Self::get_chunk_coords_at(self.camera_position);
            if current_chunk_coords != self.last_camera_chunk_coords {
                camera_moved_chunks = true;
            }
        }

        self.process_pending_uploads();

        if camera_moved_chunks {
            self.request_chunks_around_camera();
        }

        if let Some(rcx) = self.rcx.as_mut() {
            rcx.window.request_redraw();
        }
    }
}

fn window_size_dependent_setup(
    window_size: PhysicalSize<u32>,
    images: &[Arc<Image>],
    render_pass: &Arc<RenderPass>,
    memory_allocator: &Arc<StandardMemoryAllocator>,
    vs: &EntryPoint,
    fs: &EntryPoint,
) -> (Vec<Arc<Framebuffer>>, Arc<GraphicsPipeline>) {
    let device = memory_allocator.device();

    let depth_buffer = ImageView::new_default(
        Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::D16_UNORM,
                extent: images[0].extent(),
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap(),
    )
    .unwrap();

    let framebuffers = images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view, depth_buffer.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    let pipeline = {
        let vertex_input_state = VertexData::per_vertex().definition(vs).unwrap();
        let stages = [
            PipelineShaderStageCreateInfo::new(vs.clone()),
            PipelineShaderStageCreateInfo::new(fs.clone()),
        ];
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();
        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState {
                    viewports: [Viewport {
                        offset: [0.0, 0.0],
                        extent: [window_size.width as f32, window_size.height as f32],
                        depth_range: 0.0..=1.0,
                    }]
                    .into_iter()
                    .collect(),
                    ..Default::default()
                }),
                rasterization_state: Some(RasterizationState {
                    cull_mode: vulkano::pipeline::graphics::rasterization::CullMode::Back,
                    polygon_mode: PolygonMode::Fill,
                    ..Default::default()
                }),
                depth_stencil_state: Some(DepthStencilState {
                    depth: Some(DepthState::simple()),
                    ..Default::default()
                }),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap()
    };

    (framebuffers, pipeline)
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/vert.glsl",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/frag.glsl",
    }
}
