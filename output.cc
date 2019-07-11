#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/numerics/data_out.h>

#include <iostream>
#include <fstream>

#include "equation.h"
#include "claw.h"

using namespace dealii;

// @sect4{ConservationLaw::output_results}

// This function now is rather
// straightforward. All the magic, including
// transforming data from conservative
// variables to physical ones has been
// abstracted and moved into the
// EulerEquations class so that it can be
// replaced in case we want to solve some
// other hyperbolic conservation law.
//
// Note that the number of the output file is
// determined by keeping a counter in the
// form of a static variable that is set to
// zero the first time we come to this
// function and is incremented by one at the
// end of each invokation.

//--------------------------------------------------------------------------------------------
//write out the primal solution
//--------------------------------------------------------------------------------------------
template <int dim>
void ConservationLaw<dim>::output_results () const
{
   typename EulerEquations<dim>::Postprocessor
   postprocessor (parameters.schlieren_plot);
   
   DataOut<dim> data_out;
   data_out.attach_dof_handler (dof_handler);
   
   data_out.add_data_vector (current_solution,
                             EulerEquations<dim>::component_names (),
                             DataOut<dim>::type_dof_data,
                             EulerEquations<dim>::component_interpretation ());
   
   data_out.add_data_vector (current_solution, postprocessor);  //add derived_quantities
   
   data_out.build_patches (mapping(), fe.degree);
   
   static unsigned int output_file_number = 0;
   std::string filename = "solution-" + Utilities::int_to_string (output_file_number, 3);
   
   if(parameters.output_format == "vtk")     
      filename += ".vtu";
   else if(parameters.output_format == "tecplot") 
      filename += ".plt";
   else if(parameters.output_format == "gnuplot")
      filename += ".gnuplot";
   
   std::cout << "Writing file " << filename << std::endl;
   std::ofstream output (filename.c_str());
   
   if(parameters.output_format == "vtk")     
   {
      DataOutBase::VtkFlags flags(elapsed_time, output_file_number);
      data_out.set_flags(flags);
      data_out.write_vtu (output);
   }
   else if(parameters.output_format == "tecplot")  
      data_out.write_tecplot (output);
   else if(parameters.output_format == "gnuplot")
      data_out.write_gnuplot (output);

   ++output_file_number;
   
   // Write shock indicator
   DataOut<dim> shock;
   shock.attach_dof_handler (dh_cell);

   shock.add_data_vector (mu_shock, "mu_shock", DataOut<dim>::type_cell_data);
   shock.add_data_vector (shock_indicator, "shock_indicator", DataOut<dim>::type_cell_data);
   

   shock.build_patches (mapping());

   if(parameters.output_format == "vtk")     
   {
      std::ofstream shock_output ("shock.vtu");
      shock.write_vtu (shock_output);
   }
   else if(parameters.output_format == "tecplot") 
   {
      std::ofstream shock_output ("shock.plt");
      shock.write_tecplot (shock_output);
   }
   else if(parameters.output_format == "gnuplot")
   {
      std::ofstream shock_output ("shock.gnuplot");
      shock.write_gnuplot (shock_output);
   }
   
   // Write cell average solution
//   DataOut<dim> avg;
//   avg.attach_dof_handler (dof_handler0);
//   avg.add_data_vector (cell_average,
//                        EulerEquations<dim>::component_names (),
//                        DataOut<dim>::type_dof_data,
//                        EulerEquations<dim>::component_interpretation ());
//   avg.build_patches (mapping());
//   if(parameters.output_format == "vtk")     
//   {
//      std::ofstream avg_output ("avg.vtu");
//      avg.write_vtu (avg_output);
//   }
//   else if(parameters.output_format == "tecplot") 
//   {
//      std::ofstream avg_output ("avg.plt");
//      avg.write_tecplot (avg_output);
//   }

   // Write current solution Vector to a file, for use of setting initial condition from existing solution file
   std::ofstream solution_out("flow_field_solution.dat",std::ios::out);  //if file do not exist, then creat one, if exists, then clear its contents.
   solution_out.precision(15);   //must add this line in order to keep the precision
   int len = current_solution.size();
   for(int i=0;i<len;++i)
      solution_out<<current_solution[i]<<std::endl;
   solution_out.close();
}

//--------------------------------------------------------------------------------------------
// Write out refinement indicators
//--------------------------------------------------------------------------------------------
template <int dim>
void ConservationLaw<dim>::output_refinement_indicators (dealii::Vector<double> &indicator) const
{
   DataOut<dim> data_out;
   data_out.attach_dof_handler (dof_handler);
   data_out.add_data_vector (indicator, 
                             "refinement_indicator",
                             DataOut<dim>::type_cell_data);//TODO

   data_out.build_patches (mapping());

   std::cout << "Writing file of refinement_indicator" << std::endl;   
   std::ofstream output ("refinement_indicator.plt");
   data_out.write_tecplot (output);
}

//--------------------------------------------------------------------------------------------
// Write out dofs----time----estimated_error_obj successively to the same file
//--------------------------------------------------------------------------------------------
template <int dim>
void ConservationLaw<dim>::output_converge_curve (unsigned int dofs, double estimated_error_obj) const
{
   std::ofstream curve_graph("error_vs_dof.plt", std::ios::app);      
//   timer = std::clock();
//   double current_time = (double)(timer-start)/CLOCKS_PER_SEC;
   curve_graph.seekp(0, std::ios::end);
   std::fstream::off_type lon = curve_graph.tellp();
   if(lon == 0){    //the file has not been writen
       curve_graph <<"title = \"error vs dofs\""<<std::endl
                   <<"variables = \"dofs\", \"error\""<<std::endl
                   <<"zone t = \""<< parameters.strategy_name << "\"" << std::endl;
       curve_graph << dofs <<"	"<< estimated_error_obj << std::endl; 
   }
   else
       curve_graph << dofs <<"	"<< estimated_error_obj << std::endl; 
}

template class ConservationLaw<2>;
