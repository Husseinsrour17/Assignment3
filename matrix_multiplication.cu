#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define TILE_DIM 16

__global__ void multiplyMatrices(int* A, int* B, int* C, int A_rows, int A_cols, int B_cols) {
    int rowIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int colIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (rowIndex < A_rows && colIndex < B_cols) {
        int aggregate = 0;
        for (int n = 0; n < A_cols; ++n) {
            aggregate += A[rowIndex * A_cols + n] * B[n * B_cols + colIndex];
        }
        C[rowIndex * B_cols + colIndex] = aggregate;
    }
}

int main() {
    int rowsOfA, colsOfA, colsOfB;
    printf("Input dimensions of matrix A (rows): ");
    scanf("%d", &rowsOfA);
    printf("Input dimensions for matrix B (columnsA x columnsB): ");
    scanf("%d %d", &colsOfA, &colsOfB);

    int *A, *B, *C;
    int *d_A, *d_B, *d_C;

    size_t sizeOfA = rowsOfA * colsOfA * sizeof(int);
    size_t sizeOfB = colsOfA * colsOfB * sizeof(int);
    size_t sizeOfC = rowsOfA * colsOfB * sizeof(int);

    A = (int*)malloc(sizeOfA);
    B = (int*)malloc(sizeOfB);
    C = (int*)malloc(sizeOfC);

    srand((unsigned int)time(NULL));
    for (int i = 0; i < rowsOfA * colsOfA; ++i) {
        A[i] = rand() % 10;
    }
    for (int i = 0; i < colsOfA * colsOfB; ++i) {
        B[i] = rand() % 10;
    }
    cudaMalloc((void**)&d_A, sizeOfA);
    cudaMalloc((void**)&d_B, sizeOfB);
    cudaMalloc((void**)&d_C, sizeOfC);

    cudaMemcpy(d_A, A, sizeOfA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeOfB, cudaMemcpyHostToDevice);

    dim3 blockDims(TILE_DIM, TILE_DIM);
    dim3 gridDims((colsOfB + blockDims.x - 1) / blockDims.x, (rowsOfA + blockDims.y - 1) / blockDims.y);

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    cudaEventRecord(begin);

    multiplyMatrices<<<gridDims, blockDims>>>(d_A, d_B, d_C, rowsOfA, colsOfA, colsOfB);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float timeElapsed = 0;
    cudaEventElapsedTime(&timeElapsed, begin, end);

    cudaMemcpy(C, d_C, sizeOfC, cudaMemcpyDeviceToHost);

    printf("Elapsed time: %f ms\n", timeElapsed);

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);

    return 0;
}
