#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <time.h>
#include <math.h>
#include <vector>
#include <list>
#include <algorithm>
#include <numeric>
using namespace std;




int main(int argc, const char * argv[]) {
    
    int i;
    int j;
    int l_begin;
    int n_points;
    int n_faces;
    int max_vert;
    int p;
    float p_tot[3];
    int n_neig;
    int n_cell;
    int n_west;
    int start_west;
    int n_east;
    int start_east;
    int n_hill;
    int start_hill;
    
    string line;
    string a, b, c;
    stringstream line_i;
    
    // Default path:
    string batch_number = argv[1];
    string batch_direction = argv[2];
    string batch = argv[3];
    string scratch = argv[4];
    string mesh_dir = scratch+"/"+batch+"/"+batch+"_"+batch_number+"_"+batch_direction+"/simpleFoam/constant/polyMesh";
    string coord_dir = scratch+"/intel_wind/data_generation/openfoam_batch/initialization/coord";
    
    
    
    //
    // POINTS
    //
    
    ifstream points_file;
    points_file.open(mesh_dir+"/points");
    
    // Find first line
    i=0;
    while(getline(points_file, line))
    {
        i++;
        if(line=="(")
        {
            l_begin = i+1;
            break;
        }
    }
    
    points_file.clear();
    points_file.seekg(0, ios::beg);
    
    // Find number of points
    i=0;
    while(getline(points_file, line))
    {
        i++;
        if(i==l_begin-2)
        {
            n_points = stoi(line);
            break;
        }
    }
    
    points_file.clear();
    points_file.seekg(0, ios::beg);
    
    // Save coordinates of points
    vector<float> points_x(n_points);
    vector<float> points_y(n_points);
    vector<float> points_z(n_points);
    i=0;
    j=0;
    while(getline(points_file, line))
    {
        i++;
        if(i>=l_begin && i<(l_begin+n_points))
        {
            replace( line.begin(), line.end(), '(', ' ');
            replace( line.begin(), line.end(), ')', ' ');
            line_i << line;
            line_i >> a >> b >> c;
            
            if(line.find("e-") != string::npos)
            {
                if(a.find("e-") != string::npos)
                {
                    a="0";
                }
                if(b.find("e-") != string::npos)
                {
                    b="0";
                }
                if(c.find("e-") != string::npos)
                {
                    c="0";
                }
            }
            
            points_x[j] = stof(a);
            points_y[j] = stof(b);
            points_z[j] = stof(c);
            j++;
        }
    }
    
    points_file.close();
    
    
    
    
    //
    // FACES
    //
    
    ifstream faces_file;
    faces_file.open(mesh_dir+"/faces");
    
    // Find first line
    i=0;
    while(getline(faces_file, line))
    {
        i++;
        if(line=="(")
        {
            l_begin = i+1;
            break;
        }
    }
    
    faces_file.clear();
    faces_file.seekg(0, ios::beg);
    
    // Find number of faces
    i=0;
    while(getline(faces_file, line))
    {
        i++;
        if(i==l_begin-2)
        {
            n_faces = stoi(line);
            break;
        }
    }
    
    faces_file.clear();
    faces_file.seekg(0, ios::beg);
    
    // Find number of vertices for each face and maximum number of vertices
    i=0;
    j=0;
    max_vert=0;
    vector<int> n_vert(n_faces);
    while(getline(faces_file, line))
    {
        i++;
        if(i>=l_begin && j<n_faces && line != "")
        {
            
            a = line[0];
            b = line[1];
            
            if(b=="(")
            {
                n_vert[j] = stoi(a);
            }
            else
            {
                n_vert[j] = stoi(a)*10+stoi(b);
            }
            
            if(n_vert[j]>10) // se sono più di 10 li mette a capo, quindi devo saltare le prossime righe
            {
                int k = 0;
                while(getline(faces_file, line))
                {
                    k++;
                    if(k>n_vert[j]+1)
                    {
                        break;
                    }
                }
            }
            
            j++;
        }
    }
    
    max_vert = *max_element(n_vert.begin(), n_vert.end());
    
    faces_file.clear();
    faces_file.seekg(0, ios::beg);
    
    // Find coordinates of cell centroid
    i=0;
    j=0;
    vector<string> f;
    vector<float> faces_x(n_faces);
    vector<float> faces_y(n_faces);
    vector<float> faces_z(n_faces);
    
    while(getline(faces_file, line))
    {
        i++;
        if(i>=l_begin && j<n_faces && line != "")
        {
            f.clear();
            f.resize(n_vert[j]+1);
            
            p_tot[0]=0;
            p_tot[1]=0;
            p_tot[2]=0;
            
            if(n_vert[j]<11)
            {
                replace( line.begin(), line.end(), '(', ' ');
                replace( line.begin(), line.end(), ')', ' ');
                line_i << line;
                
                for(int k=0; k<n_vert[j]+1; k++)
                {
                    line_i >> f[k];
                }
            }
            else
            {
                int k = 0;
                while(getline(faces_file, line))
                {
                    f[k] = line;
                    k++;
                    
                    if(k>n_vert[j]+1)
                    {
                        break;
                    }
                }
            }
            
            for(int k=0; k<n_vert[j]; k++)
            {
                p = stoi(f[k+1]);
                p_tot[0] = p_tot[0] + points_x[p];
                p_tot[1] = p_tot[1] + points_y[p];
                p_tot[2] = p_tot[2] + points_z[p];
            }
            
            faces_x[j] = p_tot[0]/n_vert[j];
            faces_y[j] = p_tot[1]/n_vert[j];
            faces_z[j] = p_tot[2]/n_vert[j];
            
            j++;
        }
    }
    
    faces_file.close();
    
    
    
    //
    // BOUNDARY FACES
    //
    
    ifstream boundary_file;
    boundary_file.open(mesh_dir+"/boundary");
    
    // West faces
    
    i=0;
    while(getline(boundary_file, line))
    {
        i++;
        if(line.find("west_face") != string::npos)
        {
            break;
        }
    }
    while(getline(boundary_file, line))
    {
        i++;
        if(line.find("nFaces") != string::npos)
        {
            replace( line.begin(), line.end(), 'n', ' ');
            replace( line.begin(), line.end(), 'F', ' ');
            replace( line.begin(), line.end(), 'a', ' ');
            replace( line.begin(), line.end(), 'c', ' ');
            replace( line.begin(), line.end(), 'e', ' ');
            replace( line.begin(), line.end(), 's', ' ');
            n_west = stoi(line);
        }
        
        if(line.find("startFace") != string::npos)
        {
            replace( line.begin(), line.end(), 's', ' ');
            replace( line.begin(), line.end(), 't', ' ');
            replace( line.begin(), line.end(), 'a', ' ');
            replace( line.begin(), line.end(), 'r', ' ');
            replace( line.begin(), line.end(), 'F', ' ');
            replace( line.begin(), line.end(), 'c', ' ');
            replace( line.begin(), line.end(), 'e', ' ');
            start_west = stoi(line);
            break;
        }
    }
    
    boundary_file.clear();
    boundary_file.seekg(0, ios::beg);
    
    
    // East faces
    
    i=0;
    while(getline(boundary_file, line))
    {
        i++;
        if(line.find("east_face") != string::npos)
        {
            break;
        }
    }
    while(getline(boundary_file, line))
    {
        i++;
        if(line.find("nFaces") != string::npos)
        {
            replace( line.begin(), line.end(), 'n', ' ');
            replace( line.begin(), line.end(), 'F', ' ');
            replace( line.begin(), line.end(), 'a', ' ');
            replace( line.begin(), line.end(), 'c', ' ');
            replace( line.begin(), line.end(), 'e', ' ');
            replace( line.begin(), line.end(), 's', ' ');
            n_east = stoi(line);
        }
        
        if(line.find("startFace") != string::npos)
        {
            replace( line.begin(), line.end(), 's', ' ');
            replace( line.begin(), line.end(), 't', ' ');
            replace( line.begin(), line.end(), 'a', ' ');
            replace( line.begin(), line.end(), 'r', ' ');
            replace( line.begin(), line.end(), 'F', ' ');
            replace( line.begin(), line.end(), 'c', ' ');
            replace( line.begin(), line.end(), 'e', ' ');
            start_east = stoi(line);
            break;
        }
    }
    
    boundary_file.clear();
    boundary_file.seekg(0, ios::beg);
    
    
    // hill_geometry
    
    i=0;
    while(getline(boundary_file, line))
    {
        i++;
        if(line.find("hill_geometry") != string::npos)
        {
            break;
        }
    }
    while(getline(boundary_file, line))
    {
        i++;
        if(line.find("nFaces") != string::npos)
        {
            replace( line.begin(), line.end(), 'n', ' ');
            replace( line.begin(), line.end(), 'F', ' ');
            replace( line.begin(), line.end(), 'a', ' ');
            replace( line.begin(), line.end(), 'c', ' ');
            replace( line.begin(), line.end(), 'e', ' ');
            replace( line.begin(), line.end(), 's', ' ');
            n_hill = stoi(line);
        }
        
        if(line.find("startFace") != string::npos)
        {
            replace( line.begin(), line.end(), 's', ' ');
            replace( line.begin(), line.end(), 't', ' ');
            replace( line.begin(), line.end(), 'a', ' ');
            replace( line.begin(), line.end(), 'r', ' ');
            replace( line.begin(), line.end(), 'F', ' ');
            replace( line.begin(), line.end(), 'c', ' ');
            replace( line.begin(), line.end(), 'e', ' ');
            start_hill = stoi(line);
            break;
        }
    }
    
    boundary_file.clear();
    boundary_file.seekg(0, ios::beg);
    
    boundary_file.close();
    
    // Write boundary
    
    ofstream east;
    east.open(coord_dir+"/EastCoordinates_"+batch_number+"_"+batch_direction);
    for(int k=start_east; k<start_east+n_east; k++)
    {
        east<<faces_x[k]<<" "<<faces_y[k]<<" "<<faces_z[k]<<endl;
    }
    east.close();
    
    ofstream west;
    west.open(coord_dir+"/WestCoordinates_"+batch_number+"_"+batch_direction);
    for(int k=start_west; k<start_west+n_west; k++)
    {
        west<<faces_x[k]<<" "<<faces_y[k]<<" "<<faces_z[k]<<endl;
    }
    west.close();
    
    ofstream hill;
    hill.open(coord_dir+"/HillCoordinates_"+batch_number+"_"+batch_direction);
    for(int k=start_hill; k<start_hill+n_hill; k++)
    {
        hill<<faces_x[k]<<" "<<faces_y[k]<<" "<<faces_z[k]<<endl;
    }
    hill.close();
    
    
    
    //
    // OWNER
    //
    
    ifstream owner_file;
    owner_file.open(mesh_dir+"/owner");
    
    i=0;
    while(getline(owner_file, line))
    {
        i++;
        if(line=="(")
        {
            l_begin = i+1;
            break;
        }
    }
    
    owner_file.clear();
    owner_file.seekg(0, ios::beg);
    
    vector<int> owner(n_faces);
    i=0;
    j=0;
    
    while(getline(owner_file, line))
    {
        i++;
        if(i>=l_begin && i<(l_begin+n_faces))
        {
            owner[j] = stoi(line);
            j++;
        }
    }
    
    owner_file.close();
    
    
    
    //
    // NEIGHBOUR
    //
    
    ifstream neighbour_file;
    neighbour_file.open(mesh_dir+"/neighbour");
    
    i=0;
    while(getline(neighbour_file, line))
    {
        i++;
        if(line=="(")
        {
            l_begin = i+1;
            break;
        }
    }
    
    neighbour_file.clear();
    neighbour_file.seekg(0, ios::beg);
    
    i=0;
    while(getline(neighbour_file, line))
    {
        i++;
        if(i==l_begin-2)
        {
            n_neig = stoi(line);
            break;
        }
    }
    
    neighbour_file.clear();
    neighbour_file.seekg(0, ios::beg);
    
    vector<int> neighbour(n_neig);
    i=0;
    j=0;
    
    while(getline(neighbour_file, line))
    {
        i++;
        if(i>=l_begin && i<(l_begin+n_neig))
        {
            neighbour[j] = stoi(line);
            j++;
        }
    }
    
    neighbour_file.close();
    
    
    
    //
    // CELLS
    //
    
    ifstream cell_file;
    cell_file.open(mesh_dir+"/cellLevel");
    
    // Find first line
    i=0;
    while(getline(cell_file, line))
    {
        i++;
        a = line[0];
        if(a=="(")
        {
            l_begin = i+1;
            break;
        }
    }
    
    cell_file.clear();
    cell_file.seekg(0, ios::beg);
    
    // Find number of cells
    i=0;
    while(getline(cell_file, line))
    {
        i++;
        if(i==l_begin-2)
        {
            n_cell = stoi(line);
            break;
        }
    }
    
    cell_file.close();
    
    
    // Find cell centroid
    vector<float> cells_x(n_cell);
    vector<float> cells_y(n_cell);
    vector<float> cells_z(n_cell);
    
    vector<list<float> > cells_x_list(n_cell);
    vector<list<float> > cells_y_list(n_cell);
    vector<list<float> > cells_z_list(n_cell);
    
    for(int k=0; k<n_faces; k++)
    {
        cells_x_list[owner[k]].__emplace_back(faces_x[k]);
        cells_y_list[owner[k]].__emplace_back(faces_y[k]);
        cells_z_list[owner[k]].__emplace_back(faces_z[k]);
    }
    
    for(int k=0; k<n_neig; k++)
    {
        cells_x_list[neighbour[k]].__emplace_back(faces_x[k]);
        cells_y_list[neighbour[k]].__emplace_back(faces_y[k]);
        cells_z_list[neighbour[k]].__emplace_back(faces_z[k]);
    }
    
    for(int k=0; k<n_cell; k++)
    {
        cells_x[k] = accumulate(cells_x_list[k].begin(), cells_x_list[k].end(), 0.0) / cells_x_list[k].size();
        cells_y[k] = accumulate(cells_y_list[k].begin(), cells_y_list[k].end(), 0.0) / cells_y_list[k].size();
        cells_z[k] = accumulate(cells_z_list[k].begin(), cells_z_list[k].end(), 0.0) / cells_z_list[k].size();
    }
    
    
    // Save cell coordinates
    ofstream coord;
    coord.open(coord_dir+"/CellCoordinates_"+batch_number+"_"+batch_direction);
    for(int k=0; k<n_cell; k++)
    {
        coord<<cells_x[k]<<" "<<cells_y[k]<<" "<<cells_z[k]<<endl;
    }
    coord.close();
    

    cout << "End c++ "+batch_number+"_"+batch_direction << endl;
    
    return 0;
}
