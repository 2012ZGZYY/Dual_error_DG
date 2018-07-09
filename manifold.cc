#include <Sacado.hpp>
#include<cmath>
#include "manifold.h"
using namespace dealii;

//----------------------------------------------------------------------------------------------------
//This is a curve boundary manifold object used in ringleb's flow. 
//it's used on the left and right side of the domain, if necessary, can also be used on interior cells
//----------------------------------------------------------------------------------------------------
//given point in the real space, return point in the reference domain
Point<2> RinglebSide::pull_back(const Point<2>& x) const
{
   double Vmin=0.2, Vmax=1.8;   //just used as initial guess for iteration process to numerically solve V
   unsigned int imax;
   double b, rho, L, V, psi,Vpsi, Vp, tol, theta;
   imax =500;
   tol = 1.0e-15;
   
   //Initial guess, Vp=1
   Vp = 0.5*( Vmin + Vmax );   

   //Start iterations. Eq.(7.13.76). given point x, return V
   for(unsigned int i=1; i<imax; ++i){
      b   = std::sqrt(1-0.2*Vp*Vp);
      rho = std::pow(b,5);
      L   = 1./b+1./(3*b*b*b)+1./(5*rho)-0.5*std::log((1+b)/(1-b));
      V   = std::sqrt(1/2./rho/std::sqrt((x[0]-L/2)*(x[0]-L/2)+x[1]*x[1]));   // Eq.(7.11.76)
      if(std::fabs(V-Vp)<tol)
         break;   
      if(i == imax){
         std::cout<<"did not converge...sorry."<<std::endl;
         exit(0);
      }
      Vp = V;
   }
   psi  = std::sqrt(0.5/V/V-rho*(x[0]-L/2));
   Vpsi = std::sqrt(0.5-V*V*(x[0]-L/2)*rho);
   if ( std::fabs( Vpsi - 1 ) < 1.0e-15  ) 
      theta = 0.5*3.141592653589793238;         // Do not rely on `asin' here.
   else
      theta = std::asin( psi*V );
   /*std::cout<<"given real_space point, (x,y) = "<<x[0]<<","<<x[1]
            <<std::endl<<"return reference_space point, (psi, V)= "<<psi<<","<<V
            <<std::endl;*/    //used in test.
   return Point<2>(psi, theta);
}
//given point in the reference domain, return point in the real space
Point<2> RinglebSide::push_forward(const Point<2>& psi_theta) const
{
   const double psi = psi_theta[0];
   const double theta  = psi_theta[1];
   const double V   = std::sin(theta)/psi;
   const double b   = std::sqrt(1-0.2*V*V);
   const double rho = std::pow(b,5);
   const double L   = 1./b+1./(3*b*b*b)+1./(5*rho)-0.5*std::log((1+b)/(1-b)); 

   if(std::fabs(psi*V-1) < 1.0e-15){
      return Point<2> ((1./rho)*(0.5/V/V- psi*psi) + 0.5*L,        //x, Eq.(7.11.66)
                       0);                         //y, Eq.(7.11.67)), to avoid sqrt(negative) below.
      }
   else{
      return Point<2> ((1./rho)*(0.5/V/V- psi*psi) + 0.5*L,            //x, Eq.(7.11.66)
                       psi/(rho*V)*std::sqrt(1.-psi*psi*V*V));         //y, Eq.(7.11.67))
      }
}

//----------------------------------------------------------------------------------------------------
// Another Manifold
// this is used at the top boundary of rinleb's flow domain(local coordinate now is psi_V, not psi_theta)
//----------------------------------------------------------------------------------------------------
//given point in the real space, return point in the reference domain
Point<2> RinglebTop::pull_back(const Point<2>& x) const
{
   double Vmin=0.2, Vmax=1.8;   //these are just used in initial guess for iteration process to numerically solve V
   unsigned int imax;
   double b, rho, L, V, psi, Vp, tol;
   imax =500;
   tol = 1.0e-15;
   
   //Initial guess, Vp=1
   Vp = 0.5*( Vmin + Vmax );   

   //Start iterations. Eq.(7.13.76). given point x, return V
   for(unsigned int i=1; i<imax; ++i){
      b   = std::sqrt(1-0.2*Vp*Vp);
      rho = std::pow(b,5);
      L   = 1./b+1./(3*b*b*b)+1./(5*rho)-0.5*std::log((1+b)/(1-b));
      V   = std::sqrt(1/2./rho/std::sqrt((x[0]-L/2)*(x[0]-L/2)+x[1]*x[1]));   // Eq.(7.11.76)
      if(std::fabs(V-Vp)<tol)
         break;   
      if(i == imax){
         std::cout<<"did not converge...sorry."<<std::endl;
         exit(0);
      }
      Vp = V;
   }
   psi  = std::sqrt(0.5/V/V-rho*(x[0]-L/2));

   /*std::cout<<"given real_space point, (x,y) = "<<x[0]<<","<<x[1]
            <<std::endl<<"return reference_space point, (psi, V)= "<<psi<<","<<V
            <<std::endl;*/      //used in test
   return Point<2>(psi, V);
}
//given point in the reference domain, return point in the real space
Point<2> RinglebTop::push_forward(const Point<2>& psi_V) const
{
   const double psi = psi_V[0];
   const double V  = psi_V[1];
   const double b   = std::sqrt(1-0.2*V*V);
   const double rho = std::pow(b,5);
   const double L   = 1./b+1./(3*b*b*b)+1./(5*rho)-0.5*std::log((1+b)/(1-b)); 

   if(std::fabs(psi*V-1) < 1.0e-15){
      return Point<2> ((1./rho)*(0.5/V/V- psi*psi) + 0.5*L,        //x, Eq.(7.11.66)
                       0);                            //y, Eq.(7.11.67)), to avoid sqrt(negative) below.
      }
   else{
        /* std::cout<<"given reference_space point, (psi, v)  = "<<psi_V[0]<<","<<psi_V[1]
            <<std::endl<<"return real_space point, (x,y)= "<<(1./rho)*(0.5/V/V- psi*psi) + 0.5*L
            <<","<<psi/(rho*V)*std::sqrt(1.-psi*psi*V*V)
            <<std::endl;*/
      return Point<2> ((1./rho)*(0.5/V/V- psi*psi) + 0.5*L,            //x, Eq.(7.11.66)
                       psi/(rho*V)*std::sqrt(1.-psi*psi*V*V));         //y, Eq.(7.11.67))
      }
}

//----------------------------------------------------------------------------------------------------
// NACA0012 airfoil Manifold. Just project the candidate point onto the true boundary in the normal
// direction to the boundary edge. see the related documentation 
//----------------------------------------------------------------------------------------------------

//used in newton method for projection to manifold. m=dy/dx, P is the candidate point, 
//x is P's x_coordinate_value. this function will modify f and d
void NACA0012::func(const double& m, const Point<2>& P, const double& x, double& f, double& d) const 
{
   //set and mark independent variable
   Sacado::Fad::DFad<double> xd = x;
   xd.diff(0,1);           //set xd to be dof 0, in a 1-dof system.

   Sacado::Fad::DFad<double> y = naca0012(xd);
   if(P(1)<0) y=-y;        //get the corresponding y_coordinate_value

   Sacado::Fad::DFad<double> F = m * (y-P[1]) + (xd - P[0]);
   f = F.val();
   d = F.fastAccessDx(0);  //dF/d(xd)
}

//note that candidate point is calculated as the convex combination of the given points 
//by Manifold::get_intermediate_point()
Point<2> NACA0012::project_to_manifold (const std::vector<Point<2>>& surrounding_points,
                                        const Point<2>&              candidate) const
{
   const double GEOMTOL = 1.0e-13;
   Assert((surrounding_points[0][1] > -GEOMTOL && surrounding_points[1][1] >-GEOMTOL) ||
          (surrounding_points[0][1] <  GEOMTOL && surrounding_points[1][1] < GEOMTOL),
          ExcMessage("End points not on same side of airfoil"));
      
   const double dx = surrounding_points[1][0] - surrounding_points[0][0];
   const double dy = surrounding_points[1][1] - surrounding_points[0][1];
   Assert (std::fabs(dx) > GEOMTOL, ExcMessage("dx is too small"));
   const double m = dy/dx;

   //initial guess for newton. x_coordinate should be between 0 and 1.
   double x = candidate[0];
   x = std::min(x, 1.0);
   x = std::max(x, 0.0);
   double f, d;
   func(m, candidate, x, f, d);     //change f and d
   unsigned int i = 0, ITMAX = 10;
   while(std::fabs(f) > GEOMTOL)
   {
      double s = -f/d;
      while(x+s < 0 || x+s>1 ) s *= 0.5;
      x = x + s;                    //modify x until converge
      func(m, candidate, x, f, d);
      ++i;
      AssertThrow(i < ITMAX, ExcMessage("Newton did not converge"));
   }
   //now x is computed
   double y = naca0012(x);
   if(candidate[1] < 0) y = -y;
   Point<2> p(x, y);
   return p;
}


