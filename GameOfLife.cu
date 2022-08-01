#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define for_x for (int x = 0; x < w; x++)
#define for_y for (int y = 0; y < h; y++)
#define for_xy for_x for_y

#define NUM_OF_GPU_THREADS 1024
#define BLOCK_DIM_x 32
#define BLOCK_DIM_y 32

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void print_matrix(unsigned *u, int h, int w) {
    for(int i = 0; i < h+2; i++) {
        for(int j = 0; j < w+2; j++) {
            printf("%d ", u[i*(w+2) + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void init(unsigned *u, int w, int h) {
    for_xy u[y*w + x] = rand() < RAND_MAX / 10 ? 1 : 0;
}

void show(unsigned *u, int w, int h) {
    printf("\033[H");
    for_y {
        for_x printf(u[y*w + x] ? "\033[07m  \033[m" : "  ");
        printf("\033[E");
    }
    fflush(stdout);
}

void my_init(unsigned *u, int w, int h) {
    for(int y = 1; y < h+1; y++)
        for(int x = 1; x < w+1; x++) 
            u[y*w + x] = rand() < RAND_MAX / 2 ? 1 : 0;
}

void my_init_twice(unsigned *u1, unsigned* u2, int w, int h) {
    // Changing the order of loops to get more cache hits
    for(int y = 1; y < h-1; y++)
        for(int x = 1; x < w-1; x++) {
            u1[y*w + x] = rand() < RAND_MAX / 2 ? 1 : 0;
            u2[y*w + x] = u1[y*w + x];
        }
}

void my_evolve(unsigned **u, unsigned **new_p, int w, int h) {
    unsigned *univ = *u;
    unsigned *temp = *new_p;

    for(int y = 1; y < h - 1; y++) {
        for(int x = 1; x < w - 1; x ++) {
            unsigned n = univ[(y-1)*w + x-1] + univ[(y-1)*w + x] + univ[(y-1)*w + x+1] + univ[(y)*w + x-1] + univ[(y)*w + x+1] + univ[(y+1)*w + x-1] + univ[(y+1)*w + x] + univ[(y+1)*w + x+1];
            temp[y*w + x] = (n == 3 || (n == 2 && univ[y*w + x]));
        }
    }
    unsigned* t = *u;
    *u = *new_p;
    *new_p = t;
    //print_matrix(*u,w,h);
}

void my_evolve_parallel(unsigned **u, unsigned **new_p, int w, int h) {
    unsigned *univ = *u;
    unsigned *temp = *new_p;

    for(int y = 1; y < h - 1; y++) {
        for(int x = 1; x < w - 1; x ++) {
            unsigned n = univ[(y-1)*w + x-1] + univ[(y-1)*w + x] + univ[(y-1)*w + x+1] + univ[(y)*w + x-1] + univ[(y)*w + x+1] + univ[(y+1)*w + x-1] + univ[(y+1)*w + x] + univ[(y+1)*w + x+1];
            temp[y*w + x] = (n == 3 || (n == 2 && univ[y*w + x]));
        }
    }

    unsigned* t = *u;
    *u = *new_p;
    *new_p = t;
    //print_matrix(*u,w,h);
}

void evolve(unsigned *u, int w, int h) {

    unsigned* tem = (unsigned*) malloc(w*h*sizeof(unsigned));
    for_y for_x {
        int n = 0;
        for (int y1 = y - 1; y1 <= y + 1; y1++)
            for (int x1 = x - 1; x1 <= x + 1; x1++)
                if (u[((y1 + h) % h)*h + ((x1 + w) % w)]) n++;

        if (u[y*h+x]) n--;
        tem[y*h+x] = (n == 3 || (n == 2 && u[y*h+x]));
    }
    for_y for_x u[y*h+x] = 1;
    free(tem);
}

void game(unsigned *u, int w, int h, int iter) {
    for (int i = 0; i < iter; i++) {
#ifdef LIFE_VISUAL
        show(u, w, h);
#endif
        evolve(u, w, h);
#ifdef LIFE_VISUAL
        usleep(200000);
#endif
    }
}


void my_game(unsigned **u, unsigned **new_p, int w, int h, int iter) {
    for (int i = 0; i < iter; i++) {
#ifdef LIFE_VISUAL
        show(u, w, h);
#endif
        my_evolve(u, new_p, w+2, h+2);
#ifdef LIFE_VISUAL
        usleep(200000);
#endif
    }
}

__global__ void GameOfLifeKernel(unsigned *u, unsigned *t, int w, int h) {
    int index_x = blockIdx.x * (blockDim.x - 2) + threadIdx.x;
    int index_y = blockIdx.y * (blockDim.y - 2) + threadIdx.y;
    int shared_index_x = threadIdx.x;
    int shared_index_y = threadIdx.y;
    int middle_square_pos = index_y * w + index_x;

    __shared__ int shared_u[BLOCK_DIM_y][BLOCK_DIM_x];

    if((index_x) < (w) && index_y < (h)) {
        shared_u[threadIdx.y][threadIdx.x] = u[middle_square_pos];
    }

    __syncthreads();

    if((index_x) < (w-1) && index_y < (h-1)) {
        if((shared_index_x > 0) && (shared_index_x < (blockDim.x - 1)) && (shared_index_y > 0) && (shared_index_y < (blockDim.y - 1))) {
            unsigned n = shared_u[shared_index_y-1][shared_index_x-1] + shared_u[shared_index_y-1][shared_index_x] + shared_u[shared_index_y-1][shared_index_x+1] + shared_u[shared_index_y][shared_index_x-1] + shared_u[shared_index_y][shared_index_x+1] + shared_u[shared_index_y+1][shared_index_x-1] + shared_u[shared_index_y+1][shared_index_x] + shared_u[shared_index_y+1][shared_index_x+1];
            t[middle_square_pos] = (n == 3 || (n == 2 && shared_u[shared_index_y][shared_index_x]));
        }
        
    }
    
    __syncthreads();

}

void my_game_parallel(unsigned **u, unsigned **new_p, int w, int h, int iter) {

    unsigned *u_gpu, *t_gpu;
    int size, grid_height, grid_width;

    // Calculating block and grid dimensions:
    grid_height = (h+BLOCK_DIM_y-3) / (BLOCK_DIM_y-2);
    grid_width = (w+BLOCK_DIM_x-3) / (BLOCK_DIM_x-2);
    
    dim3 grid_dim(grid_height, grid_width);
    dim3 block_dim(BLOCK_DIM_y, BLOCK_DIM_x);
    
    // Allocating and copying matrix to GPU
    size = (w+2) * (h+2) * sizeof(unsigned);
    cudaMalloc(&u_gpu, size);
    cudaMemcpy(u_gpu, *u, size, cudaMemcpyHostToDevice);
    cudaMalloc(&t_gpu, size);

    cudaEvent_t start = cudaEvent_t();
	cudaEvent_t stop = cudaEvent_t();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

    // Kernel call
    for (int i = 0; i < iter; i++) {
        GameOfLifeKernel<<< grid_dim, block_dim >>>(u_gpu, t_gpu, w+2, h+2);
        cudaStreamQuery(0);
        unsigned* t = u_gpu;
        u_gpu = t_gpu;
        t_gpu = t;
    }

	// Compute elapsed time
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsed = 0.f;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("Parallel implementation execution time = %f \n", elapsed);

    

    cudaMemcpy(*u, u_gpu, size, cudaMemcpyDeviceToHost);
    cudaFree( u_gpu );
    cudaFree( t_gpu );



}

void copy_initialization_to_gold_version(unsigned *u, unsigned *u_gold, int w, int h) {
    for(int i = 0; i < h; i++) {
        for(int j = 0; j < w; j++) {
            u_gold[i*w+j] = u[(i+1)*(w+2) + (j+1)];
        }
    }
}



__global__ void my_game_cuda(int* devA, int* devB, int* devC, int n){
	// Calculate index
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if( idx < n ) devC[idx] = devA[idx] + devB[idx];
}

void compare_matrices(unsigned *u, unsigned* u_parallel, int w, int h) {
    for(int y = 1; y < h - 1; y++) 
        for(int x = 1; x < w - 1; x++) {
            if(u[y*w + x] != u_parallel[y*w + x]) {
                printf("at y = %d, x = %d", y, x);
                printf("\n\n Test FAILED \n");
                exit(-1);
            }
        }
    
    printf("\n\n Test PASSED \n");
}

int main(int c, char *v[]) {
    int w = 0, h = 0, iter = 0;
    unsigned *u;
    unsigned *u_parallel;
    unsigned *temp;
    unsigned *u_gold;

    if (c > 1) w = atoi(v[1]);
    if (c > 2) h = atoi(v[2]);
    if (c > 3) iter = atoi(v[3]);
    if (w <= 0) w = 30;
    if (h <= 0) h = 30;
    if (iter <= 0) iter = 1000;

    u_gold = (unsigned *)calloc((w) * (h), sizeof(unsigned));
    u =     (unsigned *)calloc((w+2) * (h+2), sizeof(unsigned));
    u_parallel =     (unsigned *)calloc((w+2) * (h+2), sizeof(unsigned));

    my_init_twice(u, u_parallel, w+2, h+2);
    copy_initialization_to_gold_version(u, u_gold, w, h);
    

    cudaEvent_t start = cudaEvent_t();
	cudaEvent_t stop = cudaEvent_t();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

    game(u_gold, w, h, iter);

	// Compute elapsed time
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsed = 0.f;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("Gold implementation execution time = %f \n", elapsed);
    free(u_gold);

    temp = (unsigned *)calloc((w+2) * (h+2), sizeof(unsigned));
	cudaEventRecord(start, 0);

    my_game(&u, &temp, w, h, iter);

	// Compute elapsed time
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	elapsed = 0.f;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("Sequential improved implementation execution time = %f \n", elapsed);

    cudaEventRecord(start, 0);

    // Core call
    my_game_parallel(&u_parallel, &temp, w, h, iter);

    // Compute elapsed time
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	elapsed = 0.f;
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("Full implementation execution time = %f \n", elapsed);

	// release events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

    free(temp);

    compare_matrices(u, u_parallel, w+2, h+2);

    free(u);
    free(u_parallel);
    

}
