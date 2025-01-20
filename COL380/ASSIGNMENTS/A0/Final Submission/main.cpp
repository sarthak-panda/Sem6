#include <stdio.h>
#include <iostream>//for cin and cout during debug
#include <string>//for stoi
#include <filesystem>//for path joining
//#include <vector>

using namespace std;
namespace fs = std::filesystem;
//using  vector library we could have taken the input std::vector<std::vector<int>>& matrix but instead we will use int**
void matrixMultiplyIJK(double* A, double* B, int r_A, int c_A, int c_B, double* C){
    for(int i=0;i<r_A;i++){
        for(int j=0;j<c_B;j++){
            for(int k=0;k<c_A;k++){
                C[i*c_B+j]+=A[i*c_A+k]*B[k*c_B+j];
            }
        }
    }
    return;
}
void matrixMultiplyJIK(double* A, double* B, int r_A, int c_A, int c_B, double* C){
    for(int j=0;j<c_B;j++){
        for(int i=0;i<r_A;i++){
            for(int k=0;k<c_A;k++){
                C[i*c_B+j]+=A[i*c_A+k]*B[k*c_B+j];
            }
        }
    }    
    return;
}
void matrixMultiplyKIJ(double* A, double* B, int r_A, int c_A, int c_B, double* C){
    for(int k=0;k<c_A;k++){
        for(int i=0;i<r_A;i++){
            for(int j=0;j<c_B;j++){
                C[i*c_B+j]+=A[i*c_A+k]*B[k*c_B+j];
            }
        }
    }    
    return;
}
void matrixMultiplyIKJ(double* A, double* B, int r_A, int c_A, int c_B, double* C){
    for(int i=0;i<r_A;i++){
        for(int k=0;k<c_A;k++){
            for(int j=0;j<c_B;j++){
                C[i*c_B+j]+=A[i*c_A+k]*B[k*c_B+j];
            }
        }
    }    
    return;
}
void matrixMultiplyJKI(double* A, double* B, int r_A, int c_A, int c_B, double* C){
    for(int j=0;j<c_B;j++){
        for(int k=0;k<c_A;k++){
            for(int i=0;i<r_A;i++){
                C[i*c_B+j]+=A[i*c_A+k]*B[k*c_B+j];
            }
        }
    } 
    return;
}
void matrixMultiplyKJI(double* A, double* B, int r_A, int c_A, int c_B, double* C){
    for(int k=0;k<c_A;k++){
        for(int j=0;j<c_B;j++){
            for(int i=0;i<r_A;i++){
                C[i*c_B+j]+=A[i*c_A+k]*B[k*c_B+j];
            }
        }
    }    
    return;
}
void readMatrix(double* M, fs::path input_path,fs::path file_name,int r, int c){
    fs::path fullPath_mat = input_path / file_name;
    FILE *fp = fopen(fullPath_mat.string().c_str(), "rb");
    fread(M, sizeof(double), r*c, fp);
    fclose(fp);
    return;
}
void writeMatrix(double* M, fs::path output_path,fs::path file_name,int r, int c){
    fs::path fullPath_mat = output_path / file_name;
    FILE *fp = fopen(fullPath_mat.string().c_str(), "wb");
    fwrite(M, sizeof(double), r*c, fp);
    fclose(fp);
}
int main(int argc, char* argv[]){
    if ((argc-1)!=6){
        cout<<"Insufficient Arguments"<<endl;
        return 0;
    }
    int type=stoi(argv[1]);
    int r_A=stoi(argv[2]);
    int c_A=stoi(argv[3]);
    int c_B=stoi(argv[4]);
    fs::path input_path=argv[5];
    fs::path output_path=argv[6];
    double* A=new double[r_A*c_A]();
    double* B=new double[c_A*c_B]();
    double* C=new double[r_A*c_B]();
    //now we will read matrix A from input_path/mtx_A.bin we will join the paths using os.path.join
    //we will assume that the matrix is stored in row major order
    fs::path path_matA="mtx_A.bin";
    readMatrix(A,input_path,path_matA,r_A,c_A);
    // fs::path fullPath_matA = input_path / path_matA;
    // FILE *fp = fopen(fullPath_matA.string().c_str(), "rb");
    // fread(A, sizeof(double), r_A*c_A, fp);
    // fclose(fp);
    //now we will read matrix B from input_path/mtx_B.bin we will join the paths using os.path.join
    //we will assume that the matrix is stored in row major order
    fs::path path_matB="mtx_B.bin";
    readMatrix(B,input_path,path_matB,c_A,c_B);
    // fs::path fullPath_matB = input_path / path_matB;
    // fp = fopen(fullPath_matB.string().c_str(), "rb");
    // fread(B, sizeof(double), c_A*c_B, fp);
    // fclose(fp);
    //now we will multiply the matrices based on the type
    if(type==0){
        matrixMultiplyIJK(A,B,r_A,c_A,c_B,C);
    }
    else if(type==1){
        matrixMultiplyIKJ(A,B,r_A,c_A,c_B,C);
    }
    else if(type==2){
        matrixMultiplyJIK(A,B,r_A,c_A,c_B,C);
    }
    else if(type==3){
        matrixMultiplyJKI(A,B,r_A,c_A,c_B,C);
    }
    else if(type==4){
        matrixMultiplyKIJ(A,B,r_A,c_A,c_B,C);
    }
    else if(type==5){
        matrixMultiplyKJI(A,B,r_A,c_A,c_B,C);
    }
    else{
        cout<<"Invalid Type"<<endl;
        return 0;
    }
    //now we will write the result matrix C to output_path/mtx_C.bin we will join the paths using os.path.join
    //we will assume that the matrix is stored in row major order
    fs::path path_matC="mtx_C.bin";
    writeMatrix(C,output_path,path_matC,r_A,c_B);
    // fs::path fullPath_matC = output_path / path_matC;
    // fp = fopen(fullPath_matC.string().c_str(), "wb");
    // fwrite(C, sizeof(double), r_A*c_B, fp);
    // fclose(fp);
    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}