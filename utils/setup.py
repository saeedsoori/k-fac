from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bmm_cuda',
    ext_modules=[
        CUDAExtension('bmm_cuda', [
            'bmm_cuda.cpp',],
            extra_compile_args={'cxx':['-O3','-fopenmp', '-L/home/ubuntu/magma/magma-2.5.4/lib/ -lmagma_sparse -lmagma', '-I/home/ubuntu/magma/magma-2.5.4/include' , '-I/usr/local/cuda/include', '-L/usr/local/cuda/lib64 -lcudart -lcudadevrt -lcublas -lcusparse -llapack -lblas -lpthread -lm', '-DADD_'],},
             library_dirs=['/home/ubuntu/magma/magma-2.5.4/lib'],
             libraries=['magma_sparse', 'magma'],
             extra_ldflags={'cxx': [
                          '-L/usr/local/cuda/lib64 -lcudart -lcudadevrt -lcublas -lcusparse -llapack -lblas -lpthread -lm', ], },
            ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
