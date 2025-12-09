#include "template/common.h" 

inline uint RGB32FtoRGB8( float3 c )
{
	int r = (int)(min( c.x, 1.f ) * 255);
	int g = (int)(min( c.y, 1.f ) * 255);
	int b = (int)(min( c.z, 1.f ) * 255);
	return (r << 16) + (g << 8) + b;
}

// ==============================================================================================
// STRUCTS (COMMON, BVH, AND Kdtree)
// ==============================================================================================

struct Intersection
{
	float t;			// intersection distance along ray
	float u, v;			// barycentric coordinates of the intersection
	uint instPrim;		// instance index (12 bit) and primitive index (20 bit)
};

struct Ray
{
	float3 O, D, rD;	// in OpenCL, each of these will be padded to 16 bytes
	struct Intersection hit;
};

struct Tri 
{ 
	float v0x, v0y, v0z, dummy0;
	float v1x, v1y, v1z, dummy1;
	float v2x, v2y, v2z, dummy2;
	float cx, cy, cz, dummy3;
};

struct TriEx 
{ 
	float2 uv0, uv1, uv2; 
	float N0x, N0y, N0z;
	float N1x, N1y, N1z;
	float N2x, N2y, N2z;
	float dummy;
};

// --- BVH STRUCTS ---
struct BVHNode
{
	float minx, miny, minz;
	int leftFirst;
	float maxx, maxy, maxz;
	int triCount;
};

struct TLASNode
{
	float minx, miny, minz;
	uint leftRight; // 2x16 bits
	float maxx, maxy, maxz;
	uint BLAS;
};

struct BVHInstance
{
	uint dummy1, dummy2;
	uint idx;
	float16 transform;
	float16 invTransform; // inverse transform
	uint dummy[6];
};

// --- Kdtree STRUCTS ---
typedef struct __attribute__((packed)) 
{ 
    float min[3]; 
    float max[3]; 
    uint firstChildIdx; 
    uint triCount; 
	uint   axis;          
    float  splitPos;     
} KdtreeNode;

// ==============================================================================================
// HELPER FUNCTIONS
// ==============================================================================================

// Helper for KdtreeNode unpacking
float3 get_min(KdtreeNode n) { return (float3)(n.min[0], n.min[1], n.min[2]); }
float3 get_max(KdtreeNode n) { return (float3)(n.max[0], n.max[1], n.max[2]); }

// Generic AABB Intersection (Works for both)
float IntersectAABB_Generic( float3 rayOrig, float3 rayDir, float3 rayInvDir, float3 bMin, float3 bMax, int* count)
{
	(*count)++;

    float tx1 = (bMin.x - rayOrig.x) * rayInvDir.x;
    float tx2 = (bMax.x - rayOrig.x) * rayInvDir.x;
    float tmin = min( tx1, tx2 );
    float tmax = max( tx1, tx2 );

    float ty1 = (bMin.y - rayOrig.y) * rayInvDir.y;
    float ty2 = (bMax.y - rayOrig.y) * rayInvDir.y;
    tmin = max( tmin, min( ty1, ty2 ) );
    tmax = min( tmax, max( ty1, ty2 ) );

    float tz1 = (bMin.z - rayOrig.z) * rayInvDir.z;
    float tz2 = (bMax.z - rayOrig.z) * rayInvDir.z;
    tmin = max( tmin, min( tz1, tz2 ) );
    tmax = min( tmax, max( tz1, tz2 ) );

    if (tmax >= tmin && tmax > 0) return tmin; else return 1e30f;
}

// Wrapper for BVHNode (Legacy support)
float IntersectAABB_BVH( struct Ray* ray, __global struct BVHNode* node, int* count)
{
    return IntersectAABB_Generic(ray->O, ray->D, ray->rD, 
                                 (float3)(node->minx, node->miny, node->minz), 
                                 (float3)(node->maxx, node->maxy, node->maxz), count);
}

// ==============================================================================================
// TRIANGLE INTERSECTION (TWO VERSIONS)
// ==============================================================================================

// 1. BVH Version (Takes 'struct Ray*')
void IntersectTri_BVH( struct Ray* ray, __global struct Tri* tri, const uint instPrim, int* count )
{
	(*count)++;

	float3 v0 = (float3)(tri->v0x, tri->v0y, tri->v0z);
	float3 v1 = (float3)(tri->v1x, tri->v1y, tri->v1z);
	float3 v2 = (float3)(tri->v2x, tri->v2y, tri->v2z);
	float3 edge1 = v1 - v0, edge2 = v2 - v0;
	float3 h = cross( ray->D, edge2 );
	float a = dot( edge1, h );
	if (a > -0.00001f && a < 0.00001f) return;
	float f = 1 / a;
	float3 s = ray->O - v0;
	float u = f * dot( s, h );
	if (u < 0 || u > 1) return;
	const float3 q = cross( s, edge1 );
	const float v = f * dot( ray->D, q );
	if (v < 0 || u + v > 1) return;
	const float t = f * dot( edge2, q );
	if (t > 0.0001f && t < ray->hit.t)
	{
		ray->hit.t = t;
		ray->hit.u = u;
		ray->hit.v = v;
		ray->hit.instPrim = instPrim;
	}
}

// 2. Kdtree Version (Takes raw vectors, faster/cleaner for Kdtree stack)
void IntersectTri_Kdtree( float3 rayOrig, float3 rayDir, float* minT, __global struct Tri* tri, uint instIdx, uint triIdx, int* hitInstPrim, int* count)
{
	(*count)++;

	float3 v0 = (float3)(tri->v0x, tri->v0y, tri->v0z);
	float3 v1 = (float3)(tri->v1x, tri->v1y, tri->v1z);
	float3 v2 = (float3)(tri->v2x, tri->v2y, tri->v2z);
	float3 edge1 = v1 - v0, edge2 = v2 - v0;
	float3 h = cross( rayDir, edge2 );
	float a = dot( edge1, h );
	if (a > -0.00001f && a < 0.00001f) return;
	float f = 1 / a;
	float3 s = rayOrig - v0;
	float u = f * dot( s, h );
	if (u < 0 || u > 1) return;
	const float3 q = cross( s, edge1 );
	const float v = f * dot( rayDir, q );
	if (v < 0 || u + v > 1) return;
	const float t = f * dot( edge2, q );
	if (t > 0.0001f && t < *minT)
	{
		*minT = t;
	}
}

// ==============================================================================================
// BVH LOGIC (From Original File)
// ==============================================================================================

void BVHIntersect( struct Ray* ray, uint instanceIdx, 
	__global struct Tri* tri, __global struct BVHNode* bvhNode, __global uint* triIdx, __global uint* stats )
{
	int localBoxTests = 0;
    int localTriTests = 0;
	ray->rD = (float3)( 1 / ray->D.x, 1 / ray->D.y, 1 / ray->D.z );
	__global struct BVHNode* node = &bvhNode[0], *stack[64];
	uint stackPtr = 0;
	while (1)
	{
		if (node->triCount > 0)
		{
			for (uint i = 0; i < node->triCount; i++)
			{
				uint instPrim = (instanceIdx << 20) + triIdx[node->leftFirst + i];
				IntersectTri_BVH( ray, &tri[instPrim & 0xfffff], instPrim, &localTriTests );
			}
			if (stackPtr == 0) break; else node = stack[--stackPtr];
			continue;
		}
		__global struct BVHNode* child1 = &bvhNode[node->leftFirst];
		__global struct BVHNode* child2 = &bvhNode[node->leftFirst + 1];
		float dist1 = IntersectAABB_BVH( ray, child1, &localBoxTests );
		float dist2 = IntersectAABB_BVH( ray, child2, &localBoxTests );
		if (dist1 > dist2) 
		{ 
			float d = dist1; dist1 = dist2; dist2 = d;
			__global struct BVHNode* c = child1; child1 = child2; child2 = c; 
		}
		if (dist1 == 1e30f)
		{
			if (stackPtr == 0) break; else node = stack[--stackPtr];
		}
		else
		{
			node = child1;
			if (dist2 != 1e30f) 
			{
				if (stackPtr < 64) stack[stackPtr++] = child2;
			}
		}
	}
	if (localBoxTests > 0) atomic_add(&stats[0], localBoxTests);
    if (localTriTests > 0) atomic_add(&stats[1], localTriTests);
}

float3 Trace_BVH( struct Ray* ray, __global float* skyPixels, __global struct Tri* triData, 
	__global struct BVHNode* bvhNodeData, __global uint* idxData, __global uint* stats )
{
	BVHIntersect( ray, 0, triData, bvhNodeData, idxData, stats );
	if (ray->hit.t < 1e30f) return (float3)(1,1,1);
	float phi = atan2( ray->D.z, ray->D.x );
	uint u = (uint)(3200 * (phi > 0 ? phi : (phi + 2 * PI)) * INV2PI - 0.5f);
	uint v = (uint)(1600 * acos( ray->D.y ) * INVPI - 0.5f);
	uint skyIdx = (u + v * 3200) % (3200 * 1600);
	return 0.65f * (float3)(skyPixels[skyIdx * 3], skyPixels[skyIdx * 3 + 1], skyPixels[skyIdx * 3 + 2]);
}

// ==============================================================================================
// Kdtree LOGIC (New)
// ==============================================================================================

void IntersectKdtree( 
    float3 rayOrig, float3 rayDir, float3 rayInvDir, float* dist, 
    __global KdtreeNode* nodes, __global uint* triIndices, 
    __global struct Tri* tris, uint instIdx, int* instPrim, __global uint* stats)
{
	int localBoxTests = 0;
    int localTriTests = 0;
    uint stack[64];
    uint stackPtr = 0;
    stack[stackPtr++] = 0;

    while (stackPtr > 0)
    {
        uint nodeIdx = stack[--stackPtr];
        KdtreeNode node = nodes[nodeIdx];
        
        if (node.triCount > 0)
        {
            for (uint i = 0; i < node.triCount; i++)
            {
                uint triIndex = triIndices[node.firstChildIdx + i];
                IntersectTri_Kdtree( rayOrig, rayDir, dist, &tris[triIndex], instIdx, triIndex, instPrim, &localTriTests);
            }
            continue; 
        }

       
		
		uint leftIdx  = node.firstChildIdx;
		uint rightIdx = leftIdx + 1;
				
		for (int k = 0; k < 2; k++)
		{
			uint childIdx = (k == 0) ? leftIdx : rightIdx;
			KdtreeNode child = nodes[childIdx];
		
			if (child.triCount == 0 && child.firstChildIdx == 0) continue;

		    float3 cMin = get_min(child);
		    float3 cMax = get_max(child);
			float boxDist = IntersectAABB_Generic(rayOrig, rayDir, rayInvDir, cMin, cMax, &localBoxTests);
			if (boxDist < *dist && stackPtr < 64)
				stack[stackPtr++] = childIdx;
		}
    }
	if (localBoxTests > 0) atomic_add(&stats[0], localBoxTests);
    if (localTriTests > 0) atomic_add(&stats[1], localTriTests);
}

// ==============================================================================================
// KERNELS
// ==============================================================================================

// KERNEL 1: Original BVH Render
__kernel void render_bvh( __global uint* target, __global float* skyPixels,
	__global struct Tri* triData, __global struct TriEx* triExData,
	__global uint* texData, __global struct TLASNode* tlasData,
	__global struct BVHInstance* instData,
	__global struct BVHNode* bvhNodeData, __global uint* idxData,
	float3 camPos, float3 p0, float3 p1, float3 p2, int pixelOffset,
	__global uint* stats
)
{
	int threadIdx = get_global_id( 0 ) + pixelOffset;
	if (threadIdx >= SCRWIDTH * SCRHEIGHT) return;
	int x = threadIdx % SCRWIDTH;
	int y = threadIdx / SCRWIDTH;
	struct Ray ray;
	ray.O = camPos;
	float3 pixelPos = ray.O + p0 + (p1 - p0) * ((float)x / SCRWIDTH) + (p2 - p0) * ((float)y / SCRHEIGHT);
	ray.D = normalize( pixelPos - ray.O );
	ray.hit.t = 1e30f;
	float3 color = Trace_BVH( &ray, skyPixels, triData, bvhNodeData, idxData, stats );
	target[x + y * SCRWIDTH] = RGB32FtoRGB8( color );
}

// KERNEL 2: New Kdtree Render
__kernel void render_kdtree( 
    __global uint* target, __global float* skyPixels, __global struct Tri* triData, 
    __global struct TriEx* triExData, __global uint* texData, 
    __global void* tlasData, __global void* instData, 
    __global KdtreeNode* KdtreeNodes, __global uint* KdtreeIndices, 
    float3 camPos, float3 p0, float3 p1, float3 p2, int pixelOffset,
	__global uint* stats
)
{
	int threadIdx = get_global_id( 0 ) + pixelOffset;
	if (threadIdx >= SCRWIDTH * SCRHEIGHT) return;
	int x = threadIdx % SCRWIDTH;
	int y = threadIdx / SCRWIDTH;

	float3 rayOrig = camPos;
	float3 pixelPos = rayOrig + p0 + (p1 - p0) * ((float)x / SCRWIDTH) + (p2 - p0) * ((float)y / SCRHEIGHT);
	float3 rayDir = normalize( pixelPos - rayOrig );
    float3 rayInvDir = (float3)(1.0f / rayDir.x, 1.0f / rayDir.y, 1.0f / rayDir.z);

	float t = 1e30f; 
    int instPrim = -1; 
    IntersectKdtree( rayOrig, rayDir, rayInvDir, &t, KdtreeNodes, KdtreeIndices, triData, 0, &instPrim, stats );

	float3 color;
	if (t < 1e30f) color = (float3)(1.0f, 1.0f, 1.0f);
    else
    {
        float phi = atan2( rayDir.z, rayDir.x );
        uint u = (uint)(3200 * (phi > 0 ? phi : (phi + 2 * PI)) * INV2PI - 0.5f);
        uint v = (uint)(1600 * acos( rayDir.y ) * INVPI - 0.5f);
        uint skyIdx = (u + v * 3200) % (3200 * 1600);
        color = 0.65f * (float3)(skyPixels[skyIdx * 3], skyPixels[skyIdx * 3 + 1], skyPixels[skyIdx * 3 + 2]);
    }
	target[x + y * SCRWIDTH] = RGB32FtoRGB8( color );
}