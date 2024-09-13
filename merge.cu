#include <iostream>
#include <math.h>
#include <unistd.h>
#include <string>
#include <fcntl.h>
#include <random>

using namespace std;

int write_disk(int in, void* tmp_buffer , unsigned long long size , unsigned long long file_offset)
{
    unsigned long long left_size = size;
    while(left_size > 0) {
        ssize_t return_value;
        return_value = pwrite(in, tmp_buffer, size, file_offset);
        if(return_value < 0)
            return return_value;
        left_size -= return_value;
        file_offset += return_value;
    }
    return size;
}

int check_result(int fd,unsigned long long entry_num )
{
    std::cout << "check result" << std::endl;
    
    // for(int i = 0; i < 10; i++) {
    //     pread(fd, &value1, sizeof(int), i * sizeof(int));
    //     std::cout << "fd [ "<< i << "] = "<<value1 <<std::endl;
    // }
    unsigned long long tmp;
    pread(fd, &tmp, sizeof(int), (1742863087UL)*sizeof(int));
    cout<< "value [ 1742863087]  "<<tmp<<endl;
    std::mt19937 generator;
    for(unsigned long long i = 1; i < 1000; i++) {
        unsigned long long value1;
        unsigned long long value2;
        unsigned long long index1 = generator()%(entry_num -1 ); 
        unsigned long long index2 = generator()%(index1 ); 
        pread(fd, &value1, sizeof(int), (index1)*sizeof(int));
        pread(fd, &value2, sizeof(int), (index2)*sizeof(int));
        if(value2 > value1) {
            cout << "error" << endl;
            std::cout << "value [ "<<index1 <<"] = " << value1 <<std::endl;
            std::cout << "value [ "<<index2   <<"] = " << value2 <<std::endl;
            return -1;
        }
        value1 = value2;
    }
    std::cout << " =========== check pass ============ "<< std::endl;
    return 0;
}




void block_sort(int in,  unsigned long long file_offset , unsigned long long block_size, int thread_num) {
    // read data from input file
    
    int* tmp_buffer = (int * )malloc(block_size * sizeof(int));
    std::cout << " block_size  :"<< block_size  <<std::endl;
    std::cout << " file_offset :"<< file_offset << std::endl;
    pread(in, tmp_buffer, block_size * sizeof(int), file_offset);
    //void *devPtr;
    //cudaMalloc(&devPtr, block_size * sizeof(int));
    //cudaMemcpy(devPtr, tmp_buffer, block_size * sizeof(int), cudaMemcpyHostToDevice);
    // sort data in memory
    ssize_t return_value;
    for(int i = 0; i < block_size; i++) {
        tmp_buffer[i] = i/1024;
    }
    return_value = write_disk(in, tmp_buffer, block_size * sizeof(int), file_offset);
    std::cout << " write size : " << block_size * sizeof(int) << endl;
    std::cout << " pwrite return value : "<< return_value<<endl;
    for(int i = 0; i < block_size; i++) {
        int  tmp;
        return_value = pread(in, &tmp, sizeof(int), i*sizeof(int));
        if(return_value != sizeof(int))
            std::cout << " pread return value : "<< return_value<<endl;
        if(tmp_buffer[i]!=tmp)
        {
            cout<< "start enequal!!! at "<<i<<endl;
            break;
        }
        
    }
    // cout<< "value [ 1742863087]  "<<tmp_buffer[1742863087]<<endl;
    // // write sorted data to output file
    // //cudaMemcpy(tmp_buffer, devPtr, block_size * sizeof(int), cudaMemcpyDeviceToHost);
    
    

    // cout<< "value [ 1742863087]  "<<tmp<<endl;
    free(tmp_buffer);
    check_result(in,2UL*1024*1024*1024);
}

//* this function use binary search to find the partition index in the second file
void binary_find_partition_index(int in, int partitial_num, int* partitial_value, int offset, int entry_num , int * partitial_index, int* partitial_num2){
    int cursor_num = partitial_num*2  ;
    int* cursor_value = (int*)malloc(cursor_num * sizeof(int));
    int* cursor_index = (int*)malloc(cursor_num * sizeof(int));
    for(int i = 0; i < cursor_num ; i++) {
        cursor_index[i] = i*entry_num/cursor_num;
        pread(in, &cursor_value[i], sizeof(int), offset + cursor_index[i] * sizeof(int));
    }
    //! need to be checked
    int left = 0;
    int right = entry_num - 1;
    for(int i = 0; i < partitial_num -1; i++) {  // (partitial_num -1 ) partition points
        for(int j = 0; j < cursor_num; j++) {
            if(partitial_value[i] <= cursor_value[j]) {
                right = cursor_index[j];
                if(i >0 && partitial_index[i-1] > left) {
                    left = partitial_index[i-1];
                }
                else {
                    left = cursor_index[j - 1];
                }
                break;
            }
        }
        while(left < right) {
            int mid = (left + right) / 2;
            int mid_value ;
            pread(in, &mid_value, sizeof(int), offset + mid * sizeof(int));
            if(partitial_value[i] <= mid_value) {
                right = mid;
            }
            else {
                left = mid + 1;
            }
        }
        partitial_index[i] =  left;
        if(i != 0) {
            partitial_num2[i-1] = partitial_index[i] - partitial_index[i-1];
        }

        if(i == partitial_num - 1) {
            partitial_num2[i] = entry_num - partitial_index[i];
        }
        
    }
}

__global__
void merge_kernel(int** a, int** b, int** c, int* left_num_1, int* left_num_2, int* dest_num)
{
    int block_id = blockIdx.x; //todo need to be checked
    //int start_offset = blockIdx.x * blockDim.x + threadIdx.x;  //todo need to be checked
    int index_1 =0;
    int index_2 =0;
    int index_3 =0;
    while(1){
        if(a[block_id][index_1] <= b[block_id][index_2]) {
            c[block_id][index_3] = a[block_id][index_1];
            index_1++;
            index_3++;
        }
        else {
            c[block_id][index_3] = b[block_id][index_2];
            index_2++;
            index_3++;
        }
        //* judge when to escape : if a or b is empty
        if(index_1 == left_num_1[block_id] || index_2 == left_num_2[block_id] ) {
            //* move the rest of the data to the front
            if(index_1 == left_num_1[block_id]) {
                for(int i = 0; i < left_num_2[block_id] - index_2; i++) {
                    b[block_id][i] = b[block_id][i+ index_2];
                }
            }
            else {
                for(int i = 0; i < left_num_1[block_id] - index_1; i++) {
                    a[block_id][i ] = a[block_id][i+ index_1];
                }
            }
            break;
        }
    }
    left_num_1[block_id] -= index_1;
    left_num_2[block_id] -= index_2;
    dest_num[block_id] = index_3;
}

void parallel_merge(int in, int  out, int fetch_num, int thread_num, int* partitial_index1,int* partitial_index2,int* partitial_num1 ,int* partitial_num2 ,int partitial_num,int offset ) {
    int ** p_tmpbuffer1 = (int**)malloc(partitial_num * sizeof(int*));
    for(int i = 0; i < partitial_num; i++) {
        p_tmpbuffer1[i] = (int*)malloc(fetch_num * sizeof(int));
    }

    int ** p_tmpbuffer2 = (int**)malloc(partitial_num * sizeof(int*));
    for(int i = 0; i < partitial_num; i++) {
        p_tmpbuffer2[i] = (int*)malloc(fetch_num * sizeof(int));
    }
    int ** p_destbuffer = (int**)malloc(  partitial_num * sizeof(int*));
    for(int i = 0; i <  partitial_num; i++) {
        p_destbuffer[i] = (int*)malloc(2*fetch_num * sizeof(int));
    }
    int * left_num_1 = (int*)malloc(partitial_num * sizeof(int));
    for(int i = 0; i < partitial_num; i++) {
        left_num_1[i] = partitial_num1[i];
    }
    int * left_num_2 = (int*)malloc(partitial_num * sizeof(int));
    for(int i = 0; i < partitial_num; i++) {
        left_num_2[i] = partitial_num2[i];
    }
    int * tmpbuffer1_left = (int*)malloc(partitial_num * sizeof(int));
    for(int i = 0; i < partitial_num; i++) {
        tmpbuffer1_left[i] = 0;
    }
    int * tmpbuffer2_left = (int*)malloc(partitial_num * sizeof(int));
    for(int i = 0; i < partitial_num; i++) {
        tmpbuffer2_left[i] = 0;
    }
    int * destbuffer_num = (int*)malloc(partitial_num * sizeof(int));
    for(int i = 0; ; i++) { // merge two block
        bool flag = false;
        for(int j = 0; j < partitial_num; j++) {
            if(left_num_1[j] > 0 || left_num_2[j] > 0 || tmpbuffer1_left[j] > 0 || tmpbuffer2_left[j] > 0) {
                flag = true;
            }
        }
        if(!flag) {
            break;
        }
        //* phase 1: read data from input file to each tmp buffer
        for(int j = 0; j < partitial_num; j++) {
            if(left_num_1[j] > 0 && tmpbuffer1_left[j] == 0) {
                int read_num = (left_num_1[j] > fetch_num) ? fetch_num : left_num_1[j];
                pread(in, p_tmpbuffer1[j], read_num * sizeof(int), offset + partitial_index1[j] * sizeof(int));
                left_num_1[j] -= read_num;
                tmpbuffer1_left[j] = read_num;
            }
            if(left_num_2[j] > 0 && tmpbuffer2_left[j] == 0) {
                int read_num = (left_num_2[j] > fetch_num) ? fetch_num : left_num_2[j];
                pread(in, p_tmpbuffer2[j], read_num * sizeof(int), offset + partitial_index2[j] * sizeof(int));
                left_num_2[j] -= read_num;
                tmpbuffer2_left[j] = read_num;
            }
        }
        //* phase 2: use gpu kernels parallel merge data in memory and move the unsorted data from back to front
        
        int** devPtr1;
        int** devPtr2;
        int** devPtr3;
        for(int j = 0; j < partitial_num; j++) {
            cudaMalloc(&devPtr1[j], fetch_num * sizeof(int));
            cudaMalloc(&devPtr2[j], fetch_num * sizeof(int));
            cudaMalloc(&devPtr3[j], 2*fetch_num * sizeof(int));
            cudaMemcpy(devPtr1[j], p_tmpbuffer1[j], fetch_num * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(devPtr2[j], p_tmpbuffer2[j], fetch_num * sizeof(int), cudaMemcpyHostToDevice);
        }
        int* devPtr4;
        int* devPtr5;
        int* devPtr6;
        cudaMalloc(&devPtr4, partitial_num * sizeof(int));
        cudaMalloc(&devPtr5, partitial_num * sizeof(int));
        cudaMalloc(&devPtr6, partitial_num * sizeof(int));
        cudaMemcpy(devPtr4, left_num_1, partitial_num * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(devPtr5, left_num_2, partitial_num * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(devPtr6, destbuffer_num, partitial_num * sizeof(int), cudaMemcpyHostToDevice);
        merge_kernel<<<partitial_num, fetch_num>>>(devPtr1, devPtr2, devPtr3, devPtr4, devPtr5, devPtr6);
        for(int j = 0; j < partitial_num; j++) {
            cudaMemcpy(p_tmpbuffer1[j], devPtr1[j], fetch_num * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(p_tmpbuffer2[j], devPtr2[j], fetch_num * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(p_destbuffer[j], devPtr3[j], 2*fetch_num * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(devPtr1[j]);
            cudaFree(devPtr2[j]);
            cudaFree(devPtr3[j]);
        }
        
        
        
        //* phase 3: write data to output file 
        for(int j = 0; j < partitial_num; j++) {
            if(destbuffer_num[j] > 0) {
                pwrite(out, p_destbuffer[j], destbuffer_num[j] * sizeof(int), offset + partitial_index1[j] * sizeof(int));
                partitial_index1[j] += destbuffer_num[j];
            }
        }
    }

}

void merge_two(int in, int out, int block_size, int thread_num, int offset1, int offset2, int entry_num1, int entry_num2) {
    // find partitial value in a
    int partitial_num = thread_num -1;
    int* partitial_value = (int*)malloc(partitial_num * sizeof(int));
    int* partitial_index1 = (int*)malloc(partitial_num * sizeof(int));
    int* partitial_index2 = (int*)malloc(partitial_num * sizeof(int));
    int* partitial_num1 = (int*)malloc(partitial_num * sizeof(int));
    int* partitial_num2 = (int*)malloc(partitial_num * sizeof(int));
    for(int i = 0; i < partitial_num; i++) {
        partitial_index1[i] = i * entry_num1 / partitial_num;
        partitial_value[i] = 0;
        pread(in, &partitial_value[i], sizeof(int), offset1 + partitial_index1[i] * sizeof(int));
        partitial_num1[i] = (i == partitial_num - 1) ? entry_num1 - partitial_index1[i] : partitial_index1[i+1] - partitial_index1[i];
    }
    // find partitial index in b
    binary_find_partition_index(in, partitial_num, partitial_value, offset2, entry_num2, partitial_index2, partitial_num2);
    // read data from input file

    // merge two sorted data in memory

}

void merge_pass(int in, int out, int block_num ,int thread_num, int * offset_info, int * entrynum_info, int block_size) {
    // read data from input file
    int merge_block_num = block_num  / 2;    //* if not the exponent of 2, the last block will be merged with the last block
    
    for(int i = 0; i < merge_block_num; i++) {
        merge_two(in, out, block_size, thread_num, offset_info[2*i], offset_info[2*i+1], entrynum_info[2*i], entrynum_info[2*i+1]);
    }
}


int merge_main(string fpath1, string fpath2, unsigned long long entry_num, unsigned long long block_size, int thread_num) {

    // phase 1: sort each block size data in memory 

    int fd_1;
    fd_1 = open(fpath1.c_str(), O_RDWR, 0);
    int fd_2;
    fd_2 = open(fpath2.c_str(),O_RDWR, 0);
    int block_num = (entry_num + block_size -1 )/block_size;
    for(int i = 0; i < block_num; i++) {
        if(i == block_num - 1) {
            block_sort(fd_1, i * block_size * sizeof(int), entry_num - i * block_size, thread_num);
        }
        else {
            block_sort(fd_1, i * block_size * sizeof(int), block_size, thread_num);
        }
    }
    
    //phase 2: merge sorted data
    int offset_info_even[block_num];
    int offset_info_odd[block_num];
    int entrynum_info_even[block_num];
    int entrynum_info_odd[block_num];
    for(int i = 0; i < block_num -1; i++) {
        offset_info_even[i] = i * block_size * sizeof(int);
        entrynum_info_even[i] = block_size;
    }
    offset_info_even[block_num - 1] = (block_num - 1) * block_size * sizeof(int);
    entrynum_info_even[block_num - 1] = entry_num - (block_num - 1) * block_size;
    int block_num_even = block_num;
    int block_num_odd ;
    //int block_size; //todo need to be calculated
    int pass_num = floor(log2(block_num));
    for(int i=0; i<pass_num; i++) {
        if(i % 2 == 0) {
            block_num_odd = block_num_even / 2;
            merge_pass(fd_1, fd_2, block_num_even, thread_num, offset_info_even, entrynum_info_even,block_size);
        }
        else {
            block_num_even = block_num_odd / 2;
            merge_pass(fd_2, fd_1, block_num_odd, thread_num, offset_info_odd, entrynum_info_odd,block_size);
        }
    }
    close(fd_1);
    close(fd_2);
    return 0;
}



int main(void)
{
    string s1="/home/szy/ssd1/testfile1";
    string s2="/home/szy/ssd1/testfile2";
    merge_main(s1,s2,4UL*1024*1024*1024,2UL*1024*1024*1024,128);
    int fd = open(s1.c_str(), O_RDWR, 0);
    check_result(fd,4UL*1024*1024*1024);
    return 0;
}