﻿// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>
#include<vector>
#include<deque>
#include<unordered_map>
#include<functional>
#include<vk_mem_alloc.h>

#include<vk_mesh.h>

#include<glm/glm.hpp>

struct Material {
	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;
};

struct RenderObject {
	Mesh* mesh;

	Material* material;

	glm::mat4 transformMatrix;
};

struct MeshPushConstants {
	glm::vec4 data;
	glm::mat4 render_matrix;
};


class PipelineBuilder {
public:
	std::vector<VkPipelineShaderStageCreateInfo> _shaderStages;
	VkPipelineVertexInputStateCreateInfo _vertexInputInfo;
	VkPipelineInputAssemblyStateCreateInfo _inputAssembly;
	VkViewport _viewport;
	VkRect2D _scissor;
	VkPipelineRasterizationStateCreateInfo _rasterizer;
	VkPipelineColorBlendAttachmentState _colorBlendAttachment;
	VkPipelineMultisampleStateCreateInfo _multisampling;
	VkPipelineLayout _pipelineLayout;
	VkPipelineDepthStencilStateCreateInfo _depthStencil;

	VkPipeline build_pipeline(VkDevice device, VkRenderPass pass);
};

struct DeletionQueue
{
	std::deque<std::function<void()>>deletors;

	void push_function(std::function<void()>&& function) {
		deletors.push_back(function);
	}

	void flush() {
		// reverse iterate the deletion queue to execute all the functions
		for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
			(*it)(); // call the function
		}

		deletors.clear();
	}
};

class VulkanEngine {
public:

	bool _isInitialized{ false };
	int _frameNumber{ 0 };
	int _selectedShader{ 0 };

	VkExtent2D _windowExtent{ 800,600 };

	struct SDL_Window* _window{ nullptr };

	//initializes everything in the engine
	void init();

	//shuts down the engine
	void cleanup();

	//draw loop
	void draw();

	//run main loop
	void run();



public:
	//default array of renderable objects
	std::vector<RenderObject>_renderables;

	std::unordered_map<std::string, Material>_materials;
	std::unordered_map<std::string, Mesh> _meshes;

	Material* create_material(VkPipeline pipeline, VkPipelineLayout layout, const std::string& name);

	Material* get_material(const std::string& name);

	Mesh* get_mesh(const std::string& name);

	void draw_objects(VkCommandBuffer cmd, RenderObject* first, int count);

	//--omitted --
	VkInstance _instance;
	VkDebugUtilsMessengerEXT _debug_messenger;  // Vulkan debug output handle
	VkPhysicalDevice _choseGPU;
	VkDevice _device;
	VkSurfaceKHR _surface;  // Vulkan window surface

	VkSwapchainKHR _swapchain;

	//image format expected by the windowing system
	VkFormat _swapchainImageFormat;

	//array of images from the swapchain
	std::vector<VkImage> _swapchainImages;

	//array of image-views from the swapchain
	std::vector<VkImageView> _swapchainImageViews;

	VkImageView _depthImageView;
	AllocatedImage _depthImage;

	VkFormat _depthFormat;


	VmaAllocator  _allocator;

	VkQueue _graphicsQueue;
	uint32_t _graphicsQueueFamily;

	VkCommandPool _commandPool;
	VkCommandBuffer _mainCommandBuffer;

	VkRenderPass _renderPass;

	std::vector<VkFramebuffer> _framebuffers;

	VkSemaphore _presentSemaphore, _renderSemaphore;
	VkFence _renderFence;

	VkPipelineLayout _trianglePipelineLayout;
	VkPipeline _trianglePipeline;
	VkPipeline _redTrianglePipeline;

	VkPipelineLayout _meshPipelineLayout;
	VkPipeline _meshPipeline;

	Mesh _triangleMesh;
	Mesh _monkeyMesh;

	bool load_shader_module(const char* filePath, VkShaderModule* outShaderModule);
private:
	DeletionQueue _mainDeletionQueue;

private:

	void init_vulkan();

	void init_swapchain();

	void init_commands();

	void init_default_renderpass();
	
	void init_framebuffers();

	void init_sync_structures();

	void init_pipelines();

	void load_meshes();

	void init_scene();

	void upload_mesh(Mesh& mesh);
};
