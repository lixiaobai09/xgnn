#include <iostream>
#include <unistd.h>
using namespace std;

#define CUDA_CALL(func)                         \
 {                                              \
    cudaError_t err = func;                     \
    if(err != cudaSuccess) {                    \
        cout << __FILE__ << ":" << __LINE__     \
             << " " << #func << " "             \
             << cudaGetErrorString(err)         \
             << " errnum " << err;              \
        exit(EXIT_FAILURE);                     \
    }                                           \
 }

int main() {
    int *arr[8];
    for (int i = 0; i < 8; i++) {
        CUDA_CALL(cudaSetDevice(i));
        CUDA_CALL(cudaMalloc(&arr[i], 16859004928));
    }

    while(1) {
        sleep(10);
    }
}