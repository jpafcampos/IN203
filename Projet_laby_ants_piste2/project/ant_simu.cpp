#include <vector>
#include <iostream>
#include <random>
#include "labyrinthe.hpp"
#include "ant.hpp"
#include "pheromone.hpp"
#include "gui/context.hpp"
#include "gui/colors.hpp"
#include "gui/point.hpp"
#include "gui/segment.hpp"
#include "gui/triangle.hpp"
#include "gui/quad.hpp"
#include "gui/event_manager.hpp"
#include "display.hpp"
#include <chrono>
#include <mpi.h>
  
#define TAG 0
 
void advance_time(const labyrinthe &land, pheromone &phen,
                  const position_t &pos_nest, const position_t &pos_food,
                  std::vector<ant> &ants, std::size_t &cpteur)
{
    for (size_t i = 0; i < ants.size(); ++i)
        ants[i].advance(phen, land, pos_food, pos_nest, cpteur);
    phen.do_evaporation();
    phen.update();
    //std::cout << "cpteur:" << cpteur << std::endl;
}


int main(int nargs, char *argv[])
{
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    bool fim = 0;

    MPI_Init(&nargs, &argv);

    int rank, size;
    //MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const dimension_t dims{32, 64}; // Dimension du labyrinthe
    const std::size_t life = int(dims.first * dims.second);
    const int nb_ants = 2 * dims.first * dims.second; // Nombre de fourmis

    const double alpha = 0.97; // Coefficient de chaos
    //const double beta=0.9999; // Coefficient d'évaporation
    const double beta = 0.999; // Coefficient d'évaporation
    //
    labyrinthe laby(dims);
    // Location du nid
    position_t pos_nest{dims.first / 2, dims.second / 2};
    // Location de la nourriture
    position_t pos_food{dims.first - 1, dims.second - 1};

    
    //rank responsable par les calculs
    if (rank == 1)
    {
        //initialisation
        const double eps = 0.75; // Coefficient d'exploration
        //size_t food_quantity = 0; 

        // Définition du coefficient d'exploration de toutes les fourmis.
        ant::set_exploration_coef(eps);

        // On va créer toutes les fourmis dans le nid :
        std::vector<ant> ants;
        ants.reserve(nb_ants);
        for (size_t i = 0; i < nb_ants; ++i)
            ants.emplace_back(pos_nest, life);

        // On crée toutes les fourmis dans la fourmilière.
        pheromone phen(laby.dimensions(), pos_food, pos_nest, alpha, beta);

        size_t food_quantity = 0;
        
        while(1)
        {
            //std::cout << "food :" << food_quantity << std::endl;
            std::vector <double> buffer_calc;
            //std::vector <double> ants_positions;
            //std::vector <double> phen_values;

            //getting ants positions
            for(size_t i = 0; i < nb_ants; ++i){
                position_t current_pos = ants[i].get_position();
                buffer_calc.emplace_back( (double)current_pos.first ) ;
                buffer_calc.emplace_back( (double)current_pos.second ); 
            }
            
            //getting current phen values
            for (size_t i = 0; i < laby.dimensions().first; ++i)
                for (size_t j = 0; j < laby.dimensions().second; ++j)
                    buffer_calc.emplace_back( (double)phen(i, j) );
                
            //current food quantity
            buffer_calc.emplace_back( (double)food_quantity );
            
            MPI_Request request;
            MPI_Status status;
            MPI_Isend(buffer_calc.data(), buffer_calc.size(), MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD, &request);
            advance_time(laby, phen, pos_nest, pos_food, ants, food_quantity);
            MPI_Wait(&request, &status);

            
            /*std::cout << "comida em 1 :" << food_quantity << std::endl;
            */

            if (food_quantity >= 100 && fim == 0){
                end = std::chrono::system_clock::now();
                std::chrono::duration<double> resta_seg = end - start;
                std::cout << resta_seg.count() << std::endl;
                fim = 1;
            }
        }    
    }

    //rank responsable par l'affichage
    if (rank == 0)
    {
        //std::cout << "olh" << std::endl;
        size_t buf_size = 2*nb_ants+dims.first*dims.second+1;
        pheromone phen_display (laby.dimensions(), pos_food, pos_nest, alpha, beta);
        std::vector <ant> ants_display;
        std::vector <double> buffer_display (buf_size);
        size_t food_display;
        std::vector <position_t> ants_positions;
        
        MPI_Status status;
        MPI_Recv (buffer_display.data(), buf_size, MPI_DOUBLE, 1, TAG, MPI_COMM_WORLD, &status);

        food_display = buffer_display[buffer_display.size()-1];    
        
        for(size_t i = 0; i < 2*nb_ants; i+=2){
            ants_positions.emplace_back( position_t(buffer_display[i], buffer_display[i+1]) );
        }
        
        //on crée les fourmis avec leur positions actuelles
        ants_display.reserve(nb_ants);
        for (size_t i = 0; i < nb_ants; ++i)
            ants_display.emplace_back(ants_positions[i], life);


        gui::context graphic_context(nargs, argv);
        gui::window &win = graphic_context.new_window(h_scal * laby.dimensions().second, h_scal * laby.dimensions().first + 266);

        display_t displayer(laby, phen_display, pos_nest, pos_food, ants_display, win);


        gui::event_manager manager;
        manager.on_key_event(int('q'), [](int code) { exit(0); });

        manager.on_display([&] {displayer.display(food_display);win.blit(); });

        manager.on_idle([&]() {
            //printf("aqui\n");

            displayer.display(food_display);

        
            win.blit();
            MPI_Recv (buffer_display.data(), buf_size, MPI_DOUBLE, 1, TAG, MPI_COMM_WORLD, &status);
            
            /*ants_positions.clear();

            for(size_t i = 0; i < 2*nb_ants; i+=2)
                ants_positions.emplace_back( position_t(buffer_display[i], buffer_display[i+1]) );
*/
            for (size_t i = 0, j = 0; i < 2*nb_ants; ++j, i+=2)
                ants_display[j].new_position(position_t(buffer_display[i], buffer_display[i+1]) );
            
            food_display = buffer_display[buffer_display.size()-1];  
            
            //on met a jour la valeur des pheromones
            phen_display.change_phen_values(std::vector<double>(buffer_display.begin()+2*nb_ants, buffer_display.end()-1 ));
  
        
            //std::cout << "blit\n";
           

        });
        manager.loop();
    }
    return 0;
}
