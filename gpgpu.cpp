#include "precomp.h"
#include "bvh.h"
#include "gpgpu.h"

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
	Timer t;
	t.reset();
	//mesh = new Mesh("assets/teapot.obj", "assets/bricks.png", 3); // 3072 verts
	//mesh = new Mesh("assets/bunny.obj", "assets/bricks.png", 3); // 14904 verts
	mesh = new Mesh("assets/dragon.obj", "assets/bricks.png", 3); // 57996 verts
	//mesh = new Mesh("assets/human.obj", "assets/bricks.png", 3); // 146754 verts
	//mesh = new Mesh("assets/mustang.obj", "assets/bricks.png", 3); // 3000000 verts

	printf("Scene Stats: %d Triangles, %d Vertices\n", mesh->triCount, mesh->triCount * 3);

	for (int i = 0; i < 16; i++)
		bvhInstance[i] = BVHInstance( mesh->bvh, i );
	tlas = TLAS( bvhInstance, 16 );
	tlas.Build();
	buildTime = t.elapsed() * 1000.0f;

	memoryUsage = mesh->bvh->nodesUsed * sizeof(BVHNode) 
				+ mesh->triCount * sizeof(uint)
				+ tlas.nodesUsed * sizeof(TLASNode);

	printf("Build Time: %.2f ms\n", buildTime);
	printf("Memory Usage: %zu Bytes\n", memoryUsage);
	// load HDR sky
	skyPixels = stbi_loadf( "assets/sky_19.hdr", &skyWidth, &skyHeight, &skyBpp, 0 );
	for (int i = 0; i < skyWidth * skyHeight * 3; i++) skyPixels[i] = sqrtf( skyPixels[i] );
	// prepare OpenCL
	tracer = new Kernel( "cl/kernels.cl", "render_bvh" );
	target = new Buffer( SCRWIDTH * SCRHEIGHT * 4 );
	skyData = new Buffer( skyWidth * skyHeight * 3 * sizeof( float ), skyPixels );
	skyData->CopyToDevice();

	uint initialStats[2] = { 0, 0 };
    statsData = new Buffer( 2 * sizeof( uint ), initialStats );
    statsData->CopyToDevice();

	triData = new Buffer( mesh->triCount * sizeof( Tri ), mesh->tri );
	triExData = new Buffer( mesh->triCount * sizeof( TriEx ), mesh->triEx );
	Surface* tex = mesh->texture;
	texData = new Buffer( tex->width * tex->height * sizeof( uint ), tex->pixels );
	instData = new Buffer( 256 * sizeof( BVHInstance ), bvhInstance );

	int tlasSize = tlas.nodesUsed * sizeof( TLASNode );
	if (tlasSize == 0) tlasSize = 4;
	tlasData = new Buffer( tlasSize, tlas.tlasNode );

	bvhData = new Buffer( mesh->bvh->nodesUsed * sizeof( BVHNode ), mesh->bvh->bvhNode );
	idxData = new Buffer( mesh->triCount * sizeof( uint ), mesh->bvh->triIdx );
	triData->CopyToDevice();
	triExData->CopyToDevice();
	texData->CopyToDevice();
	instData->CopyToDevice();
	bvhData->CopyToDevice();
	idxData->CopyToDevice();
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
	AnimateScene();
	tlasData->CopyToDevice();
	// handle input
	if (keyLeft) yaw -= 0.001f * deltaTime;
	if (keyRight) yaw += 0.001f * deltaTime;
	if (keyUp) pitch -= 0.001f * deltaTime;
	if (keyDown) pitch += 0.001f * deltaTime;
	mat4 M1 = mat4::RotateY( yaw );
	mat4 M2 = M1 * mat4::RotateX( pitch );
	float3 forward = TransformVector( make_float3( 0, 0, 1 ), M1 );
	float3 right = TransformVector( make_float3( 1, 0, 0 ), M1 );
	if (keyW) camPos += forward * 0.01f * deltaTime;
	if (keyS) camPos -= forward * 0.01f * deltaTime;
	if (keyA) camPos -= right * 0.01f * deltaTime;
	if (keyD) camPos += right * 0.01f * deltaTime;
	if (keySpace) camPos.y += 0.01f * deltaTime;
	if (keyShift) camPos.y -= 0.01f * deltaTime;
	// setup screen plane in world space
	float ar = (float)SCRWIDTH / SCRHEIGHT;
	p0 = TransformPosition( make_float3( -1 * ar, 1, 1.5f ), M2 );
	p1 = TransformPosition( make_float3( 1 * ar, 1, 1.5f ), M2 );
	p2 = TransformPosition( make_float3( -1 * ar, -1, 1.5f ), M2 );

	//Clear Stats
	uint clearStats[2] = { 0, 0 };
    statsData->hostBuffer = clearStats;
    statsData->CopyToDevice();

	// render the scene using the GPU
	int totalPixels = SCRWIDTH * SCRHEIGHT;
	int chunks = 32;
	int pixelsPerChunk = totalPixels / chunks;
	for (int i = 0; i < chunks; i++)
	{
		int offset = i * pixelsPerChunk;
		tracer->SetArguments( 
			target, skyData, 
			triData, triExData, texData, tlasData, instData, bvhData, idxData, 
			camPos, p0, p1, p2, offset, statsData 
		);
		tracer->Run( pixelsPerChunk );
	}
	// obtain the rendered result
	target->CopyFromDevice();
	statsData->CopyFromDevice();
    
    uint* counts = statsData->GetHostPtr();
    float currentFps = (deltaTime > 0) ? 1000.0f / deltaTime : 0;
	fpsHistory.push_back(currentFps);

	if (fpsHistory.size() >= 500)
	{
		float sum = 0;
		for (float f : fpsHistory) sum += f;
		float avgFps = sum / fpsHistory.size();

		std::vector<float> sortedFps = fpsHistory;
		std::sort(sortedFps.begin(), sortedFps.end());
		float medianFps = sortedFps[sortedFps.size() / 2];

		printf("Stats [500 Frames] -> Avg FPS: %.1f | Median FPS: %.1f | AABB Tests: %u | Tri Tests: %u | Build: %.2f ms | Mem: %.2f Bytes\n",
			avgFps,
			medianFps,
			counts[0],
			counts[1],
			buildTime,
			(float)memoryUsage
		);

		fpsHistory.clear();
    }
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