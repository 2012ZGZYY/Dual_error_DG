//Written by zeng @Beihang University. If you have any question, contact me: 154462188@qq.com
//Brief Introduction:
This is the code being a part of my master thesis.
I make use of the open-source finite element code deal.II and Praveen C's DG solver dflo to 
generate an adaptivity refined dg solver. The refinement strategy is based on the so-called adjoint
method (also called dual method). 
Compared to the original solver dflo, I have added some modules and made many small changes in my
code. The main features of my code include:
1. a dual solver and refine strategy
2. high order boundary approximation of NACA airfoil
3. add several tests 
4. artificial viscosity modification

This project is NOT finished yet(2018/8). I'm working on the adjoint consistency these days. 

哎，项目烂尾了。从毕业季找工作，到正式工作，近一年来一直很忙，且由于工作换了方向，也没有太大心思搞这个了。这个版本也是当时为了找工作才上传到github上充门面的，姑且留在这给有需要的人吧。这次上传了当时看的一些书和文献。
//How to run the code:

Set environment variable DEAL_II_DIR=/path/to/dealii/installation

 To compile 

cmake .
make

 By default, it compiles debug version. To compile release version do 

make release 

This will switch to release mode and subsequent compilations will be in release mode. You can switch back to debug mode by 

make debug


