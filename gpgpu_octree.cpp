#include "precomp.h"
#include "bvh.h"
#include "gpgpu.h"
#include "octree.h"

// THIS SOURCE FILE:
// Code for the article "How to Build a BVH", part 9: GPGPU.
// This version shows how to render a scene using ray tracing on the
// GPU - without hardware ray tracing. The scene (and accstruc) is
// fully maintained on the CPU.
// Feel free to copy this code to your own framework. Absolutely no
// rights are reserved. No responsibility is accepted either.
// For updates, follow me on twitter: @j_bikker.

TheApp* CreateApp() { return new GPGPUApp(); }

// GPGPUApp implementation

void GPGPUApp::Init()
{
	mesh = new Mesh( "assets/dragon.obj", "assets/bricks.png", 3 );
	octree = new Octree();
	octree->Build(mesh);
	// load HDR sky
	skyPixels = stbi_loadf( "assets/sky_19.hdr", &skyWidth, &skyHeight, &skyBpp, 0 );
	for (int i = 0; i < skyWidth * skyHeight * 3; i++) skyPixels[i] = sqrtf( skyPixels[i] );

	unsigned int nodeSize = (unsigned int)(octree->GetNodeCount() * sizeof(OctreeNode));
	unsigned int idxSize = (unsigned int)(octree->GetTriIndexCount() * sizeof(uint));

	if (nodeSize == 0) nodeSize = 4; // Safety fallback
	if (idxSize == 0) idxSize = 4;   // Safety fallback
	// prepare OpenCL
	// gpgpu_octree.cpp -> Init()
	tracer = new Kernel( "cl/kernels.cl", "render_octree" );
	target = new Buffer( SCRWIDTH * SCRHEIGHT * 4 ); // intermediate screen buffer / render target
	skyData = new Buffer( skyWidth * skyHeight * 3 * sizeof( float ), skyPixels );
	skyData->CopyToDevice();
	triData = new Buffer( mesh->triCount * sizeof( Tri ), mesh->tri );
	triExData = new Buffer( mesh->triCount * sizeof( TriEx ), mesh->triEx );
	Surface* tex = mesh->texture;
	texData = new Buffer( tex->width * tex->height * sizeof( uint ), tex->pixels );
	//instData = new Buffer(sizeof(int));
	//tlasData = new Buffer(sizeof(int));
	octreeData = new Buffer(nodeSize, (void*)octree->GetNodes());
	octreeIdxData = new Buffer(idxSize, (void*)octree->GetTriIndices());
	triData->CopyToDevice();
	triExData->CopyToDevice();
	texData->CopyToDevice();
	//tlasData->CopyToDevice();
	//instData->CopyToDevice();
	octreeData->CopyToDevice();
	octreeIdxData->CopyToDevice();
}

void GPGPUApp::AnimateScene()
{
	// animate the scene
	static float a[16] = { 0 }, h[16] = { 5, 4, 3, 2, 1, 5, 4, 3 }, s[16] = { 0 };
	for (int i = 0, x = 0; x < 4; x++) for (int y = 0; y < 4; y++, i++)
	{
		mat4 R, T = mat4::Translate( (x - 1.5f) * 2.5f, 0, (y - 1.5f) * 2.5f );
		if ((x + y) & 1) R = mat4::RotateY( a[i] );
		else R = mat4::Translate( 0, h[i / 2], 0 );
		if ((a[i] += (((i * 13) & 7) + 2) * 0.005f) > 2 * PI) a[i] -= 2 * PI;
		if ((s[i] -= 0.01f, h[i] += s[i]) < 0) s[i] = 0.2f;
		mat4 transform = T * R * mat4::Scale( 1.5f );
		bvhInstance[i].SetTransform( transform );
	}
	// update the TLAS
	tlas.Build();
}
  
void GPGPUApp::Tick( float deltaTime )
{
	// update the TLAS
	//AnimateScene();
	//tlasData->CopyToDevice();
	// handle input
	if (keyLeft) yaw -= 0.003f * deltaTime;
	if (keyRight) yaw += 0.003f * deltaTime;
	if (keyUp) pitch -= 0.003f * deltaTime;
	if (keyDown) pitch += 0.003f * deltaTime;
	mat4 M1 = mat4::RotateY( yaw );
	mat4 M2 = M1 * mat4::RotateX( pitch );
	float3 forward = TransformVector( make_float3( 0, 0, 1 ), M1 );
	float3 right = TransformVector( make_float3( 1, 0, 0 ), M1 );
	if (keyW) camPos += forward * 0.5f * deltaTime;
	if (keyS) camPos -= forward * 0.5f * deltaTime;
	if (keyA) camPos -= right * 0.5f * deltaTime;
	if (keyD) camPos += right * 0.5f * deltaTime;
	if (keySpace) camPos.y += 0.5f * deltaTime;
	if (keyShift) camPos.y -= 0.5f * deltaTime;
	// setup screen plane in world space
	float ar = (float)SCRWIDTH / SCRHEIGHT;
	p0 = TransformPosition( make_float3( -1 * ar, 1, 1.5f ), M2 );
	p1 = TransformPosition( make_float3( 1 * ar, 1, 1.5f ), M2 );
	p2 = TransformPosition( make_float3( -1 * ar, -1, 1.5f ), M2 );
	// render the scene using the GPU
	int totalPixels = SCRWIDTH * SCRHEIGHT;
	int chunks = 32;
	int pixelsPerChunk = totalPixels / chunks;
	for (int i = 0; i < chunks; i++)
	{
		int offset = i * pixelsPerChunk;
		tracer->SetArguments( 
			target, skyData, 
			triData, triExData, texData, triData, triData, octreeData, octreeIdxData, 
			camPos, p0, p1, p2, offset 
		);
		tracer->Run( pixelsPerChunk );
	}
	// obtain the rendered result
	target->CopyFromDevice();
	memcpy( screen->pixels, target->GetHostPtr(), target->size );
}

void GPGPUApp::KeyUp( int key )
{
	if (key == GLFW_KEY_W) keyW = false;
	if (key == GLFW_KEY_A) keyA = false;
	if (key == GLFW_KEY_S) keyS = false;
	if (key == GLFW_KEY_D) keyD = false;
	if (key == GLFW_KEY_UP) keyUp = false;
	if (key == GLFW_KEY_DOWN) keyDown = false;
	if (key == GLFW_KEY_LEFT) keyLeft = false;
	if (key == GLFW_KEY_RIGHT) keyRight = false;
	if (key == GLFW_KEY_SPACE) keySpace = false;
	if (key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT) keyShift = false;
}

void GPGPUApp::KeyDown( int key )
{
	if (key == GLFW_KEY_W) keyW = true;
	if (key == GLFW_KEY_A) keyA = true;
	if (key == GLFW_KEY_S) keyS = true;
	if (key == GLFW_KEY_D) keyD = true;
	if (key == GLFW_KEY_UP) keyUp = true;
	if (key == GLFW_KEY_DOWN) keyDown = true;
	if (key == GLFW_KEY_LEFT) keyLeft = true;
	if (key == GLFW_KEY_RIGHT) keyRight = true;
	if (key == GLFW_KEY_SPACE) keySpace = true;
	if (key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT) keyShift = true;
}

// EOF