#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

double eval(std::vector<double> poly, double val)
{
   int degree = poly.size() - 1;

   double ans = 0;

   for (int i=degree; i>=0; i--)
   {
      ans += pow(val, i) * poly[i];
   }

   return ans;
}

std::vector<double> generatePoints(double start, double end, int numPoints)
{  
   std::vector<double> points(numPoints);

   double length = end - start;
   double spacing = length / (numPoints - 1);

   for (int i=0; i<numPoints; i++)
   {
      points[i] = i * spacing + start;
   }

   return points;
}

bool checkRoot(std::vector<double> roots, double point)
{  
   int size = roots.size();

   for (int i=0; i<size; i++)
   {
      if ((std::abs(roots[i] - point)) < 0.001)
      {  
         return false;
      }
   }

   if (!std::isnan(point)) return true;
   else return false;
}

int main() 
{
   std::cout << "Enter the degree of the polynomial: ";
   int degree;
   std::cin >> degree;

   std::vector<double> poly(degree + 1);

   for (int i=degree; i>=0; i--)
   {
      std::cout << "Enter coefficient " << i << ": ";
      std::cin >> poly[i];
   }

   std::cout << "Enter the start of the domain: ";
   double start;
   std::cin >> start;
   std::cout << "Enter the end of the domain: ";
   double end;
   std::cin >> end;

   std::vector<double> deriv(degree);

   for (int i=degree; i>0; i--)
   {
      deriv[i-1] = poly[i] * i;
   }

   int numPoints = 1000;
   std::vector<double> points = generatePoints(start, end, numPoints);

   int maxIter = 1000;
   
   for (int iter=0; iter<maxIter; iter++)
   {
      for (int i=0; i<numPoints; i++)
      {
         double f = eval(poly, points[i]);
         double fprime = eval(deriv, points[i]);
         points[i] = points[i] - (f / fprime);
      }
   }

   std::vector<double> roots;
   bool inRoots = false;

   for (int i=0; i<(numPoints - 1); i++)
   {
      if (roots.empty())
      {
         roots.push_back(points[i]);
      }
      else
      {  
         if (checkRoot(roots, points[i]))
         {
            roots.push_back(points[i]);
         }
      }
   }

   std::sort(roots.begin(),roots.end());

   std::cout << "Roots found:" << std::endl;
   for (int j=0; j<roots.size(); j++)
   {
      std::cout << roots[j] << std::endl;
   }

   return 0;
}