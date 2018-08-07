r = 1.0;  // cylinder radius
R = 100.0; // outer boundary radius

//n1 = 16; // points around the cylinder
//n2 = 32;
n1 = 16;
n2 = 56;

p  = 1.1;

Point(1) = { 0, 0, 0};
Point(2) = {-r, 0, 0};
Point(3) = { 0, r, 0};
Point(4) = { r, 0, 0};
Point(5) = { 0,-r, 0};

Point(6) = {-R, 0, 0};
Point(7) = { 0, R, 0};
Point(8) = { R, 0, 0};
Point(9) = { 0,-R, 0};

Line(1) = {6,2};
Circle(2) = {2,1,3};
Line(3) = {3,7};
Circle(4) = {7,1,6};

Line Loop(1) = {1,2,3,4};
Ruled Surface(1) = {1};
Transfinite Surface(1) = {6,2,3,7};

// second quadrant
Circle(5) = {3,1,4};
Line(6) = {4,8};
Circle(7) = {8,1,7};

Line Loop(2) = {5,6,7,-3};
Ruled Surface(2) = {2};
Transfinite Surface(2) = {3,4,8,7};

// third quadrant
Circle(8) = {4,1,5};
Line(9) = {5,9};
Circle(10) = {9,1,8};

Line Loop(3) = {8,9,10,-6};
Ruled Surface(3) = {3};
Transfinite Surface(3) = {4,5,9,8};

//fourth quadrant
Circle(11) = {5,1,2};
Circle(12) = {9,1,6};

Line Loop(4) = {-9,11,-1,-12};
Ruled Surface(4) = {4};
Transfinite Surface(4) = {5,9,6,2};

Recombine Surface(1);
Recombine Surface(2);
Recombine Surface(3);
Recombine Surface(4);

Transfinite Line{2,5,8,11,7,10,12,4} = n1;
Transfinite Line{6,9,-1,3} = n2 Using Progression p;

Physical Line(1) = {2,5,8,11}; // cylinder surface
Physical Line(2) = {4,7,10,12}; // outer boundary
Physical Surface(3) = {1,2,3,4}; 

