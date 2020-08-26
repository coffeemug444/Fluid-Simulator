#include <SFML/Graphics.hpp>
#include <stdlib.h> // malloc, free etc.
#include <math.h>   // floor
#include <iostream>

#define IX(x, y) ((x) + (y) * N)

#define SCREEN_SIZE 400


struct FluidCell
{
public:
    int size;   // how many cells in the square grid
    float dt;   // timestep
    float diff; // diffusion amount
    float visc; // viscosity
    
    float *s;   
    float *density;
    
    // velocities
    float *Vx;
    float *Vy;
    
    // previous velocities
    float *Vx0;
    float *Vy0;
}; typedef struct FluidCell FluidCell;


FluidCell *FluidCellCreate(int size, int diffusion, int viscosity, float dt) {
    FluidCell *cell = (FluidCell*)malloc(sizeof(*cell));
    int N = size;
    
    cell->size = size;
    cell ->dt = dt;
    cell->diff = diffusion;
    cell->visc = viscosity;
    
    cell->s = (float*)calloc(N * N, sizeof(float));
    cell->density = (float*)calloc(N * N, sizeof(float));
    
    cell->Vx = (float*)calloc(N * N, sizeof(float));
    cell->Vy = (float*)calloc(N * N, sizeof(float));
    
    cell->Vx0 = (float*)calloc(N * N, sizeof(float));
    cell->Vy0 = (float*)calloc(N * N, sizeof(float));
    
    
    return cell;
}

void FluidCellFree(FluidCell *cell) {
    free(cell->s);
    free(cell->density);
    
    free(cell->Vx);
    free(cell->Vy);
    
    free(cell->Vx0);
    free(cell->Vy0);
    
    free(cell);
}

void FluidCellAddDensity(FluidCell *cell, int x, int y, float amount) {
    int N = cell->size;
    cell->density[IX(x, y)] += amount;
}

void FluidCellAddVelocity(FluidCell *cell, int x, int y, float amountX, float amountY) {
    int N = cell->size;
    int index = IX(x, y);
    
    cell->Vx[index] += amountX;
    cell->Vy[index] += amountY;
}

static void set_bnd(int b, float *x, int N)
{
    for(int i = 1; i < N - 1; i++) {
        x[IX(i, 0  )] = b == 2 ? -x[IX(i, 1  )] : x[IX(i, 1  )];
        x[IX(i, N-1)] = b == 2 ? -x[IX(i, N-2)] : x[IX(i, N-2)];
    }
    for(int j = 1; j < N - 1; j++) {
        x[IX(0  , j)] = b == 1 ? -x[IX(1  , j)] : x[IX(1  , j)];
        x[IX(N-1, j)] = b == 1 ? -x[IX(N-2, j)] : x[IX(N-2, j)];
    }
    
    x[IX(0, 0)] = 0.5 * (x[IX(1, 0)] + x[IX(0, 1)]);
	x[IX(0, N-1)] = 0.5 * (x[IX(1, N-1)] + x[IX(0, N-2)]);
	x[IX(N-1, 0)] = 0.5 * (x[IX(N-2, 0)] + x[IX(N-1, 1)]);
    x[IX(N-1, N-1)] = 0.5 * (x[IX(N-2, N-1)] + x[IX(N-1, N-2)]);
}

static void lin_solve(int b, float *x, float *x0, float a, float c, int iter, int N)
{
    float cRecip = 1.0 / c;
    for (int k = 0; k < iter; k++) {
        for (int j = 1; j < N - 1; j++) {
            for (int i = 1; i < N - 1; i++) {
                x[IX(i, j)] =
                    (x0[IX(i, j)]
                        + a*(    x[IX(i+1, j  )]
                                +x[IX(i-1, j  )]
                                +x[IX(i  , j+1)]
                                +x[IX(i  , j-1)]
                       )) * cRecip;
            }
        }
        set_bnd(b, x, N);
    }
}

static void diffuse (int b, float *x, float *x0, float diff, float dt, int iter, int N)
{
    float a = dt * diff * (N - 2) * (N - 2);
    lin_solve(b, x, x0, a, 1 + 4 * a, iter, N);
}

static void project(float *velocX, float *velocY, float *p, float *div, int iter, int N)
{
    for (int j = 1; j < N - 1; j++) {
        for (int i = 1; i < N - 1; i++) {
            div[IX(i, j)] = -0.5f*(
                     velocX[IX(i+1, j  )]
                    -velocX[IX(i-1, j  )]
                    +velocY[IX(i  , j+1)]
                    -velocY[IX(i  , j-1)]
                )/N;
            p[IX(i, j)] = 0;
        }
    }
    set_bnd(0, div, N); 
    set_bnd(0, p, N);
    lin_solve(0, p, div, 1, 4, iter, N);
    
    for (int j = 1; j < N - 1; j++) {
        for (int i = 1; i < N - 1; i++) {
            velocX[IX(i, j)] -= 0.5f * (  p[IX(i+1, j)]
                                         -p[IX(i-1, j)]) * N;
            velocY[IX(i, j)] -= 0.5f * (  p[IX(i, j+1)]
                                         -p[IX(i, j-1)]) * N;
        }
    }
    set_bnd(1, velocX, N);
    set_bnd(2, velocY, N);
}

static void advect(int b, float *d, float *d0,  float *velocX, float *velocY, float dt, int N)
{
    float i0, i1, j0, j1;
    
    float dtx = dt * (N - 2);
    float dty = dt * (N - 2);
    
    float s0, s1, t0, t1;
    float tmp1, tmp2, x, y;
    
    float Nfloat = N;
    float ifloat, jfloat;
    int i, j;
    
    for(j = 1, jfloat = 1; j < N - 1; j++, jfloat++) { 
        for(i = 1, ifloat = 1; i < N - 1; i++, ifloat++) {
            tmp1 = dtx * velocX[IX(i, j)];
            tmp2 = dty * velocY[IX(i, j)];
            x    = ifloat - tmp1; 
            y    = jfloat - tmp2;
            
            if(x < 0.5f) x = 0.5f; 
            if(x > Nfloat + 0.5f) x = Nfloat + 0.5f; 
            i0 = floor(x); 
            i1 = i0 + 1.0f;
            if(y < 0.5f) y = 0.5f; 
            if(y > Nfloat + 0.5f) y = Nfloat + 0.5f; 
            j0 = floorf(y);
            j1 = j0 + 1.0f; 
            
            s1 = x - i0; 
            s0 = 1.0f - s1; 
            t1 = y - j0; 
            t0 = 1.0f - t1;
            
            int i0i = i0;
            int i1i = i1;
            int j0i = j0;
            int j1i = j1;
            
            d[IX(i, j)] = s0 * ( t0 * d0[IX(i0i, j0i)]
                              +( t1 * d0[IX(i0i, j1i)]))
                         +s1 * ( t0 * d0[IX(i1i, j0i)]
                              +( t1 * d0[IX(i1i, j1i)]));
        }
    }
    set_bnd(b, d, N);
}


void FluidCellStep(FluidCell *cell)
{
    int N          = cell->size;
    float visc     = cell->visc;
    float diff     = cell->diff;
    float dt       = cell->dt;
    float *Vx      = cell->Vx;
    float *Vy      = cell->Vy;
    float *Vx0     = cell->Vx0;
    float *Vy0     = cell->Vy0;
    float *s       = cell->s;
    float *density = cell->density;
    
    diffuse(1, Vx0, Vx, visc, dt, 4, N);
    diffuse(2, Vy0, Vy, visc, dt, 4, N);
    
    project(Vx0, Vy0, Vx, Vy, 4, N);
    
    advect(1, Vx, Vx0, Vx0, Vy0, dt, N);
    advect(2, Vy, Vy0, Vx0, Vy0, dt, N);
    
    project(Vx, Vy, Vx0, Vy0, 4, N);
    
    diffuse(0, s, density, diff, dt, 4, N);
    advect(0, density, s, Vx, Vy, dt, N);
}

int clamp (int val, int min, int max) {
    if (val > max) return max;
    if (val < min) return min;
    return val;
}



int main()
{
    sf::RenderWindow window(sf::VideoMode(SCREEN_SIZE, SCREEN_SIZE), "fluid");
    int framerate = 144;
    window.setFramerateLimit(framerate);
    
    
    int N = 100;
    float diff = .0;
    float visc = .0;
    
    FluidCell *fluidCell = FluidCellCreate(N, diff, visc, 0.006344);
    
    bool firstMouse = true;
    sf::Vector2i prevMouse;
    
    
    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }
        
        if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
            sf::Vector2i pos = sf::Mouse::getPosition(window);
            if (firstMouse) {
                firstMouse = false;
                prevMouse = pos;
            } else {
                FluidCellAddVelocity(fluidCell, (pos.x * N) / SCREEN_SIZE, (pos.y * N) / SCREEN_SIZE, clamp(pos.x - prevMouse.x, -10, 10), clamp(pos.y - prevMouse.y, -10, 10));
                prevMouse = pos;
            }
            
            
            FluidCellAddDensity(fluidCell, (pos.x * N) / SCREEN_SIZE, (pos.y * N) / SCREEN_SIZE, 200);
            
        }

        window.clear();
        FluidCellStep(fluidCell);
        
        
        int pix = SCREEN_SIZE / N;
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                sf::RectangleShape rec;
                rec.setSize(sf::Vector2f(pix, pix));
                rec.setPosition(sf::Vector2f(i * pix, j * pix));
                float b = fluidCell->density[IX(i, j)];
                //fluidCell->density[IX(i, j)] *= 0.999;
                if (b > 255) b = 255;
                rec.setFillColor(sf::Color(b, b, b));
                window.draw(rec);
            }
        }
        
        window.display();
    }

    return 0;
}
