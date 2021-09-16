#include <iostream>
#include <fstream>
#include "solver.cpp"
using namespace std;

int main()
{
    solver s = solver("../../Input/rule_tiny.csv");
    s.solve("../../Input/transactions_tiny.csv", "../../Output/transactions_tiny_output.csv");
    // solver s = solver("../Input/rule_2M.csv");
    // s.solve("../Input/transactions_0.csv", "output/transactions_0_output.csv");
    // s.solve("../Input/transactions_1.csv", "output/transactions_1_output.csv");
    // s.solve("../Input/transactions_2.csv", "output/transactions_2_output.csv");
    // s.solve("../Input/transactions_3.csv", "output/transactions_3_output.csv");
    // s.solve("../Input/transactions_4.csv", "output/transactions_4_output.csv");
    // s.solve("../Input/transactions_5.csv", "output/transactions_5_output.csv");
    // s.solve("../Input/transactions_6.csv", "output/transactions_6_output.csv");
    // s.solve("../Input/transactions_7.csv", "output/transactions_7_output.csv");
    // s.solve("../Input/transactions_8.csv", "output/transactions_8_output.csv");
    // s.solve("../Input/transactions_9.csv", "output/transactions_9_output.csv");
    return 0;
}