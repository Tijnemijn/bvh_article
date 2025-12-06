#pragma once

namespace Tmpl8
{

	// Forward declaration so the compiler knows 'Octree' is a class
	class Octree;

	// application class
	class GPGPUApp : public TheApp
	{
	public:
		// game flow methods
		void Init();
		void AnimateScene();
		float3 Trace(Ray& ray, int rayDepth = 0);
		void Tick(float deltaTime);
		void Shutdown() { /* implement if you want to do something on exit */ }
		// input handling
		void MouseUp(int button) { /* implement if you want to detect mouse button presses */ }
		void MouseDown(int button) { /* implement if you want to detect mouse button presses */ }
		void MouseMove(int x, int y) { mousePos.x = x, mousePos.y = y; }
		void MouseWheel(float y) { /* implement if you want to handle the mouse wheel */ }
		void KeyUp(int key);
		void KeyDown(int key);
		// data members
		int2 mousePos;
		Mesh* mesh;

		Octree* octree;

		BVHInstance bvhInstance[256];
		TLAS tlas;
		float3 p0, p1, p2;	// virtual screen plane corners
		float* skyPixels;
		int skyWidth, skyHeight, skyBpp;
		Kernel* tracer;		// the ray tracing kernel
		Buffer* target;		// buffer encapsulating texture that holds the rendered image
		Buffer* skyData;	// buffer for the skydome texture
		Buffer* triData;	// buffer for the mesh Tri data (vertices for intersection)
		Buffer* triExData;	// buffer for the mesh TriEx data (vertices for shading)
		Buffer* texData;	// buffer for the brick texture
		Buffer* tlasData;	// buffer to store the TLAS
		Buffer* instData;	// buffer for BVHInstance data
		Buffer* bvhData;	// buffer for BVH node data
		Buffer* idxData;	// buffer for triangle index data for BVH

		Buffer* octreeData;     // Buffer for Octree nodes
		Buffer* octreeIdxData;  // Buffer for Octree triangle indices

		// camera
		float3 camPos = float3(0, 0, -2);
		float yaw = 0, pitch = 0;
		bool keyW = false, keyA = false, keyS = false, keyD = false;
		bool keyUp = false, keyDown = false, keyLeft = false, keyRight = false;
		bool keySpace = false, keyShift = false;

		// Stats
		Buffer* statsData;		//buffer to transfer counts from GPU
		float buildTime = 0;	//build time in ms
		size_t memoryUsage = 0; // memory usage in bytes
	};

} // namespace Tmpl8