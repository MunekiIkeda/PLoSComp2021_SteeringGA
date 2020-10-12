// ***************************************
// A very simple example program for CTRNN
// ***************************************

#include "NervousSystem.h"
#include "TSearch.h"
#include "random.h"
#include "VectorMatrix.h"

#include <iostream>
#include <iomanip>
#include <fstream>   // ifstream, ofstream
#include <string>    // useful for reading and writing
#include <sstream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <vector>
#include <iomanip>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>

#define SIMULATION
#define MIGRATIONUP

#define DISPLAYANIMAL
//#define DISPLAYINDEX
#define DISPLAYFITNESS
//#define OUTPUTNEURON
//#define OUTPUTXY
//#define PRINTTOFILE

using namespace std;

// Global constants
const int numberofanimals = 100;
const double RunDuration = 1800;
const double StepSize = 0.1;
const int stepsize = 10; // 1/StepSize
const int step = 1; // StepSize/0.1
#ifdef MIGRATIONUP
const double Tcent = 17;
const double speed = 0.2;
#else
const double Tcent = 23;
const double speed = 0.3;
#endif
const double tosc = 4.2; // CPG frequency

// Search settings
const int VectSize = 23; // Size of genotype (VC)
const int PopulationSize = 96; // 96-100
const int MaxGenerations = 300; // 300
const double MutationVariance = 0.05; // 0.05

double IndexBiased[1][30], IndexControl[1][30], CurveProfile[1][6], RevturnFreq[1][6];
double FreqHigh[3][24], FreqMid[3][24], FreqLow[3][24];
double ProbHigh[38][51], ProbMid[38][51], ProbLow[38][51];
double TimeDispersion[3][8];
double RESfunc[1][100 * stepsize];
double Best[1][VectSize];

double AFDestimate(double hf_TEMP[1][100 * stepsize], double RESfunc[1][100 * stepsize])
{
	double est[1][100 * stepsize];
	double afd_est;

	for (int i = 0; i < 100 * stepsize; i++) {
		est[0][i] = hf_TEMP[0][i] * RESfunc[0][i];
	}
	afd_est = 0;
	for (int k = 0; k < 100 * stepsize; k++) {
		afd_est += est[0][k];
	}
	return afd_est;
}

double CalculateIndex(double x)
{
	int index;

	if (x <= -51) { index = 1; }
	else if (x > -51 && x <= -34) { index = 2; }
	else if (x > -34 && x <= -17) { index = 3; }
	else if (x > -17 && x <= 0) { index = 4; }
	else if (x > 0 && x <= 17) { index = 5; }
	else if (x > 17 && x <= 34) { index = 6; }
	else if (x > 34 && x <= 51) { index = 7; }
	else if (x > 51) { index = 8; }

	return index;
}

double Reflection(double x, double y, double phi)
{
	if (x > 68) { phi = M_PI - phi; }
	else if (x < -68) { phi = M_PI - phi; }
	else if (y > 48) { phi = -phi; }
	else if (y < -48) { phi = -phi; }
	return phi;
}

double XReflection(double x)
{
	if (x > 68) { x = 136 - x; }
	else if (x < -68) { x = -136 - x; }
	return x;
}

double YReflection(double y)
{
	if (y > 48) { y = 96 - y; }
	else if (y < -48) { y = -96 - y; }
	return y;
}

double newpath(double temp, int T, int state, int path, double phi, RandomState &rs)
{
	double P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11;
	double p;
	p = rs.UniformRandom(0, 1);

	if (temp > Tcent + 1.5) {
		P0 = ProbHigh[13 * T + path - 1][13 * ((state / 2) - 1)];
		P1 = P0 + ProbHigh[13 * T + path - 1][13 * ((state / 2) - 1) + 1];
		P2 = P1 + ProbHigh[13 * T + path - 1][13 * ((state / 2) - 1) + 2];
		P3 = P2 + ProbHigh[13 * T + path - 1][13 * ((state / 2) - 1) + 3];
		P4 = P3 + ProbHigh[13 * T + path - 1][13 * ((state / 2) - 1) + 4];
		P5 = P4 + ProbHigh[13 * T + path - 1][13 * ((state / 2) - 1) + 5];
		P6 = P5 + ProbHigh[13 * T + path - 1][13 * ((state / 2) - 1) + 6];
		P7 = P6 + ProbHigh[13 * T + path - 1][13 * ((state / 2) - 1) + 7];
		P8 = P7 + ProbHigh[13 * T + path - 1][13 * ((state / 2) - 1) + 8];
		P9 = P8 + ProbHigh[13 * T + path - 1][13 * ((state / 2) - 1) + 9];
		P10 = P9 + ProbHigh[13 * T + path - 1][13 * ((state / 2) - 1) + 10];
		P11 = P10 + ProbHigh[13 * T + path - 1][13 * ((state / 2) - 1) + 11];
	}
	else if (temp < Tcent - 1.5) {
		P0 = ProbLow[13 * T + path - 1][13 * ((state / 2) - 1)];
		P1 = P0 + ProbLow[13 * T + path - 1][13 * ((state / 2) - 1) + 1];
		P2 = P1 + ProbLow[13 * T + path - 1][13 * ((state / 2) - 1) + 2];
		P3 = P2 + ProbLow[13 * T + path - 1][13 * ((state / 2) - 1) + 3];
		P4 = P3 + ProbLow[13 * T + path - 1][13 * ((state / 2) - 1) + 4];
		P5 = P4 + ProbLow[13 * T + path - 1][13 * ((state / 2) - 1) + 5];
		P6 = P5 + ProbLow[13 * T + path - 1][13 * ((state / 2) - 1) + 6];
		P7 = P6 + ProbLow[13 * T + path - 1][13 * ((state / 2) - 1) + 7];
		P8 = P7 + ProbLow[13 * T + path - 1][13 * ((state / 2) - 1) + 8];
		P9 = P8 + ProbLow[13 * T + path - 1][13 * ((state / 2) - 1) + 9];
		P10 = P9 + ProbLow[13 * T + path - 1][13 * ((state / 2) - 1) + 10];
		P11 = P10 + ProbLow[13 * T + path - 1][13 * ((state / 2) - 1) + 11];
	}
	else {
		P0 = ProbMid[13 * T + path - 1][13 * ((state / 2) - 1)];
		P1 = P0 + ProbMid[13 * T + path - 1][13 * ((state / 2) - 1) + 1];
		P2 = P1 + ProbMid[13 * T + path - 1][13 * ((state / 2) - 1) + 2];
		P3 = P2 + ProbMid[13 * T + path - 1][13 * ((state / 2) - 1) + 3];
		P4 = P3 + ProbMid[13 * T + path - 1][13 * ((state / 2) - 1) + 4];
		P5 = P4 + ProbMid[13 * T + path - 1][13 * ((state / 2) - 1) + 5];
		P6 = P5 + ProbMid[13 * T + path - 1][13 * ((state / 2) - 1) + 6];
		P7 = P6 + ProbMid[13 * T + path - 1][13 * ((state / 2) - 1) + 7];
		P8 = P7 + ProbMid[13 * T + path - 1][13 * ((state / 2) - 1) + 8];
		P9 = P8 + ProbMid[13 * T + path - 1][13 * ((state / 2) - 1) + 9];
		P10 = P9 + ProbMid[13 * T + path - 1][13 * ((state / 2) - 1) + 10];
		P11 = P10 + ProbMid[13 * T + path - 1][13 * ((state / 2) - 1) + 11];
	}

	if (p <= P0) { //path = 1;
		phi = rs.UniformRandom(0, M_PI / 6);
	}
	else if (p <= P1 && p > P0) { //path = 2;
		phi = rs.UniformRandom(M_PI / 6, 2 * M_PI / 6);
	}
	else if (p <= P2 && p > P1) { //path = 3;
		phi = rs.UniformRandom(2 * M_PI / 6, 3 * M_PI / 6);
	}
	else if (p <= P3 && p > P2) { //path = 4;
		phi = rs.UniformRandom(3 * M_PI / 6, 4 * M_PI / 6);
	}
	else if (p <= P4 && p > P3) { //path = 5;
		phi = rs.UniformRandom(4 * M_PI / 6, 5 * M_PI / 6);
	}
	else if (p <= P5 && p > P4) { //path = 6;
		phi = rs.UniformRandom(5 * M_PI / 6, 6 * M_PI / 6);
	}
	else if (p <= P6 && p > P5) { //path = 7;
		phi = rs.UniformRandom(11 * M_PI / 6, 12 * M_PI / 6);
	}
	else if (p <= P7 && p > P6) { //path = 8;
		phi = rs.UniformRandom(10 * M_PI / 6, 11 * M_PI / 6);
	}
	else if (p <= P8 && p > P7) { //path = 9;
		phi = rs.UniformRandom(9 * M_PI / 6, 10 * M_PI / 6);
	}
	else if (p <= P9 && p > P8) { //path = 10;
		phi = rs.UniformRandom(8 * M_PI / 6, 9 * M_PI / 6);
	}
	else if (p <= P10 && p > P9) { //path = 11;
		phi = rs.UniformRandom(7 * M_PI / 6, 8 * M_PI / 6);
	}
	else if (p > P10) { //path = 12;
		phi = rs.UniformRandom(6 * M_PI / 6, 7 * M_PI / 6);
	}

	return phi;
}

double evaluate(TVector<double> &v, RandomState &rs) {
	// Set up the circuit
	NervousSystem n(5); // (NumberofNeurons, MaximumOutgoingChemical, MaximumOutgoingElectrical)
	double biasA = MapSearchParameter(v[1], -15, 15); // AFD
	double bias1 = MapSearchParameter(v[2], -15, 15); n.SetNeuronBias(1, bias1); // AIB
	double bias2 = MapSearchParameter(v[3], -15, 15); n.SetNeuronBias(2, bias2); // AIY
	double bias3 = MapSearchParameter(v[4], -15, 15); n.SetNeuronBias(3, bias3); // AIZ
	double bias4 = MapSearchParameter(v[5], -15, 15); n.SetNeuronBias(4, bias4); // DMN
	double bias5 = MapSearchParameter(v[6], -15, 15); n.SetNeuronBias(5, bias5); // VMN
	double weightA2 = MapSearchParameter(v[7], -15, 15); 
	double weight12 = MapSearchParameter(v[8], -15, 15); n.SetChemicalSynapseWeight(1, 2, weight12);
	double weight14 = MapSearchParameter(v[9], -15, 15); n.SetChemicalSynapseWeight(1, 4, weight14);
	double weight23 = MapSearchParameter(v[10], -15, 15); n.SetChemicalSynapseWeight(2, 3, weight23);
	double weight31 = MapSearchParameter(v[11], -15, 15); n.SetChemicalSynapseWeight(3, 1, weight31);
	double weight34 = MapSearchParameter(v[12], -15, 15); n.SetChemicalSynapseWeight(3, 4, weight34);
	double weight35 = MapSearchParameter(v[13], -15, 15); n.SetChemicalSynapseWeight(3, 5, weight35);
	double weight44 = MapSearchParameter(v[14], -10, 10); n.SetChemicalSynapseWeight(4, 4, weight44);
	double weight45 = MapSearchParameter(v[15], -15, 15); n.SetChemicalSynapseWeight(4, 5, weight45);
	double weight54 = MapSearchParameter(v[16], -15, 15); n.SetChemicalSynapseWeight(5, 4, weight54);
	double weight55 = MapSearchParameter(v[17], -10, 10); n.SetChemicalSynapseWeight(5, 5, weight55);
	double conductanceA1 = MapSearchParameter(v[18], 0, 3);
	double weightOSC = MapSearchParameter(v[19], 0, 15);
	double weightNMJ = MapSearchParameter(v[20], 0, M_PI/2);
	double Tthr = MapSearchParameter(v[21], 14, 26); 
	double Kd = MapSearchParameter(v[22], 10, 100);
	double h = MapSearchParameter(v[23], 1, 10);

	double TEMP[1][100 * stepsize], hf_TEMP[1][100 * stepsize], temp;
	double r, rt, x, y;
	int state;
	double V = speed*StepSize;
	double phi, theta, dv;
	int DV, sign;
	double AFD_state, AFD_output, CPG_output, AFD_step;
	double curve, Curve, direction_in, Direction_in;
	double f, f2, F0, F1, F2, F3, F4, F5;
	int T, path, P;
	double Tconst, Tamp, Tramp, Tstep; // Added on 20190521
	double Time;
	double Index = 0;
	int Count = 0;
	double ttxindex[numberofanimals][30], TTXindex[1][30];
	double array[2 * stepsize + 1][5];
	double vector_warm[] = { 1, 0 };
	double vect_in, vect_out;
	double curve_0_30, curve_30_60, curve_60_90, curve_90_120, curve_120_150, curve_150_180;
	double revturn_0_30, revturn_30_60, revturn_60_90, revturn_90_120, revturn_120_150, revturn_150_180;
	int n_0_30, n_30_60, n_60_90, n_90_120, n_120_150, n_150_180;
	double curve_profile[1][6], revturn_freq[1][6];
	double IndexDiff = 0;
	double CurveDiff = 0;
	double RevturnDiff = 0;
	double FitnessIndex = 1;
	double FitnessCurve = 1;
	double FitnessRevturn = 1;
	double Fitness;
	double xy_all[1800][2 * numberofanimals];


#ifdef SIMULATION
#ifdef OUTPUTNEURON
	ofstream ofs("Neurons.csv");
#endif
#ifdef OUTPUTXY
	ofstream XY("xy.csv");
#endif
	ofstream TTX("TTXIndex.csv");
	ofstream ProfileCurve("ProfileCurve.csv");
#endif
	curve_0_30 = 0; revturn_0_30 = 0; n_0_30 = 0;
	curve_30_60 = 0; revturn_30_60 = 0; n_30_60 = 0;
	curve_60_90 = 0; revturn_60_90 = 0; n_60_90 = 0;
	curve_90_120 = 0; revturn_90_120 = 0; n_90_120 = 0;
	curve_120_150 = 0; revturn_120_150 = 0; n_120_150 = 0;
	curve_150_180 = 0; revturn_150_180 = 0; n_150_180 = 0;

	for (int N = 1; N <= numberofanimals; N++) {
		// Initialization of worm state
		n.RandomizeCircuitState(-0.5, 0.5, rs); // State of circuit
		r = rs.UniformRandom(0, 5); // Radius away from center
		rt = rs.UniformRandom(0, 2 * M_PI); // Angle of position inside circle
		x = r*cos(rt); // Calculate the x position
		if (N % 3 == 1) { y = r*sin(rt); } // Three different pools in y
		else if (N % 3 == 2) { y = r*sin(rt) + 24; }
		else if (N % 3 == 0) { y = r*sin(rt) - 24; }
		state = 0;
		phi = rs.UniformRandom(0, 2 * M_PI); // Direction of the worm
		theta = rs.UniformRandom(0, 2 * M_PI); // Phase of CPG
		dv = rs.UniformRandom(-1, 1); // Dorsal or Ventral
		DV = (dv> 0) - (dv < 0);

		for (double time = StepSize; time < RunDuration + StepSize; time += StepSize) {
			if (time == StepSize) {
				for (int i = 0; i < 2 * stepsize + 1; i++) {
					array[i][0] = x;
					array[i][1] = y;
				}
				for (int i = 0; i < 100 * stepsize; i++) {
					TEMP[0][i] = Tcent + 3 * x / 68;
					if (TEMP[0][i] < Tthr) { hf_TEMP[0][i] = 0; }
					else { hf_TEMP[0][i] = pow(TEMP[0][i] - Tthr, h) / (Kd + pow(TEMP[0][i] - Tthr, h)); }
				}
				temp = Tcent + 3 * x / 68;
			}
			else {
				for (int i = 0; i < 100 * stepsize - 1; i++) {
					hf_TEMP[0][i] = hf_TEMP[0][i + 1];
				}
				temp = Tcent + 3 * x / 68;
				TEMP[0][100 * stepsize - 1] = temp;
				if (temp < Tthr) { hf_TEMP[0][100 * stepsize - 1] = 0; }
				else { hf_TEMP[0][100 * stepsize - 1] = pow(temp - Tthr, h) / (Kd + pow(temp - Tthr, h)); }
			}

			// Run the circuit
			AFD_state = AFDestimate(hf_TEMP, RESfunc);
			n.SetNeuronExternalInput(1, conductanceA1*(AFD_state - n.NeuronState(1)));
			AFD_output = sigmoid(AFD_state + biasA);
			n.SetNeuronExternalInput(2, weightA2*AFD_output);
			CPG_output = sin(2 * M_PI*(time / tosc) + theta);
			n.SetNeuronExternalInput(4, weightOSC*CPG_output);
			n.SetNeuronExternalInput(5, -weightOSC*CPG_output);
			n.EulerStep(StepSize);
			curve = weightNMJ*(n.NeuronOutput(4) - n.NeuronOutput(5));
			f = rs.UniformRandom(0, 600);
			f2 = rs.UniformRandom(0, 600);
			T = floor(time / 600);
			if (phi > 2 * M_PI) { phi = phi - 2 * M_PI; }
			else if (phi < 0) { phi += 2 * M_PI; }
			path = floor(phi / (M_PI / 6)) + 1;
			P = path;
			if (P == 7) { path = 12; P = 6; }
			else if (P == 8) { path = 11; P = 5; }
			else if (P == 9) { path = 10; P = 4; }
			else if (P == 10) { path = 9; P = 3; }
			else if (P == 11) { path = 8; P = 2; }
			else if (P == 12) { path = 7; P = 1; }
			if (temp > Tcent + 1.5) {
				F0 = FreqHigh[T][P - 1];
				F1 = F0 + FreqHigh[T][P + 5];
				F2 = F1 + FreqHigh[T][P + 11];
				F3 = F2 + FreqHigh[T][P + 17];
				F4 = F0 + FreqHigh[T][P + 17];
				F5 = 600 * FreqHigh[T][P + 5] / (FreqHigh[T][P + 5] + FreqHigh[T][P + 11]);
			}
			else if (temp < Tcent - 1.5) {
				F0 = FreqLow[T][P - 1];
				F1 = F0 + FreqLow[T][P + 5];
				F2 = F1 + FreqLow[T][P + 11];
				F3 = F2 + FreqLow[T][P + 17];
				F4 = F0 + FreqLow[T][P + 17];
				F5 = 600 * FreqLow[T][P + 5] / (FreqLow[T][P + 5] + FreqLow[T][P + 11]);
			}
			else {
				F0 = FreqMid[T][P - 1];
				F1 = F0 + FreqMid[T][P + 5];
				F2 = F1 + FreqMid[T][P + 11];
				F3 = F2 + FreqMid[T][P + 17];
				F4 = F0 + FreqMid[T][P + 17];
				F5 = 600 * FreqMid[T][P + 5] / (FreqMid[T][P + 5] + FreqMid[T][P + 11]);
			}

			if (state == 0) {
				if (f <= F0 && time < RunDuration - round(TimeDispersion[2][0])) { // omega turn
					state = 2;
					Time = time;
					phi = newpath(temp, T, state, path, phi, rs);
					x += StepSize*(TimeDispersion[T][state - 1] * cos(phi) / round(TimeDispersion[T][state - 2]));
					y += StepSize*(TimeDispersion[T][state - 1] * sin(phi) / round(TimeDispersion[T][state - 2]));
				}
				else if (f <= F1 && f > F0 && time < RunDuration - round(TimeDispersion[2][2])) { // reversal
					state = 4;
					Time = time;
					phi = newpath(temp, T, state, path, phi, rs);
					x += StepSize*(TimeDispersion[T][state - 1] * cos(phi) / round(TimeDispersion[T][state - 2]));
					y += StepSize*(TimeDispersion[T][state - 1] * sin(phi) / round(TimeDispersion[T][state - 2]));
				}
				else if (f <= F2 && f > F1 && time < RunDuration - round(TimeDispersion[2][4])) { // reverasl turn
					state = 6;
					Time = time;
					phi = newpath(temp, T, state, path, phi, rs);
					x += StepSize*(TimeDispersion[T][state - 1] * cos(phi) / round(TimeDispersion[T][state - 2]));
					y += StepSize*(TimeDispersion[T][state - 1] * sin(phi) / round(TimeDispersion[T][state - 2]));
				}
				else if (f <= F3 && f > F2 && time < RunDuration - round(TimeDispersion[2][6])) { // shallow turn
					state = 8;
					Time = time;
					phi = newpath(temp, T, state, path, phi, rs);
					x += StepSize*(TimeDispersion[T][state - 1] * cos(phi) / round(TimeDispersion[T][state - 2]));
					y += StepSize*(TimeDispersion[T][state - 1] * sin(phi) / round(TimeDispersion[T][state - 2]));
				}
				else if (f > F3) { // curve
					state = 0;
					phi += DV*curve*StepSize;
					x += V*cos(phi);
					y += V*sin(phi);
				}
			}
			else {
				if (time - Time < round(TimeDispersion[T][state - 2])) { // migration during turns
					x += StepSize*(TimeDispersion[T][state - 1] * cos(phi) / round(TimeDispersion[T][state - 2]));
					y += StepSize*(TimeDispersion[T][state - 1] * sin(phi) / round(TimeDispersion[T][state - 2]));
				}
				else { state = 0; Time = 0; }
			}
			phi = Reflection(x, y, phi);
			x = XReflection(x);
			y = YReflection(y);

			for (int i = 0; i < 2 * stepsize; i++) {
				for (int j = 0; j < 5; j++) {
					array[i][j] = array[i + 1][j];
				}
			}
			array[2 * stepsize][0] = x;
			array[2 * stepsize][1] = y;
			array[2 * stepsize][2] = temp;
			array[2 * stepsize][3] = state;
			array[2 * stepsize][4] = curve;
			double Vector_in[] = { array[2 * stepsize][0] - array[stepsize][0], array[2 * stepsize][1] - array[stepsize][1] };
			Direction_in = acos((vector_warm[0] * Vector_in[0] + vector_warm[1] * Vector_in[1]) / sqrt(pow(vector_warm[0], 2) + pow(vector_warm[1], 2)) / sqrt(pow(Vector_in[0], 2) + pow(Vector_in[1], 2))) * 180 / M_PI;
			double vector_in[] = { array[stepsize][0] - array[0][0], array[stepsize][1] - array[0][1] };
			double vector_out[] = { array[2 * stepsize][0] - array[stepsize][0], array[2 * stepsize][1] - array[stepsize][1] };
			direction_in = acos((vector_warm[0] * vector_in[0] + vector_warm[1] * vector_in[1]) / sqrt(pow(vector_warm[0], 2) + pow(vector_warm[1], 2)) / sqrt(pow(vector_in[0], 2) + pow(vector_in[1], 2))) * 180 / M_PI;
			vect_in = sqrt(pow(vector_in[0], 2) + pow(vector_in[1], 2));
			vect_out = sqrt(pow(vector_out[0], 2) + pow(vector_out[1], 2));
			double vector_change[] = { vector_out[0] / vect_out - vector_in[0] / vect_in, vector_out[1] / vect_out - vector_in[1] / vect_in };
			if (direction_in <= 45 || direction_in > 135) {
				if (vector_in[0] * vector_in[1] * vector_change[1] >= 0) { sign = -1; }
				else { sign = 1; }
			}
			else {
				if (vector_change[0] > 0) { sign = 1; }
				else { sign = -1; }
			}
			array[stepsize][4] = sign*fabs(array[stepsize][4]) * 180 / M_PI;
			Curve = sign*fabs(array[2 * stepsize][4]) * 180 / M_PI;
			if (array[stepsize][2] >= Tcent - 1.5 && array[stepsize][2] <= Tcent + 1.5) {
				if (direction_in >= 0 && direction_in < 30) {
					if (array[stepsize][3] == 0) { curve_0_30 += array[stepsize][4]; n_0_30++; }
					if (array[stepsize][3] == 0 && array[stepsize + 1][3] == 6) { revturn_0_30++; }
				}
				else if (direction_in >= 30 && direction_in < 60) {
					if (array[stepsize][3] == 0) { curve_30_60 += array[stepsize][4]; n_30_60++; }
					if (array[stepsize][3] == 0 && array[stepsize + 1][3] == 6) { revturn_30_60++; }
				}
				else if (direction_in >= 60 && direction_in < 90) {
					if (array[stepsize][3] == 0) { curve_60_90 += array[stepsize][4]; n_60_90++; }
					if (array[stepsize][3] == 0 && array[stepsize + 1][3] == 6) { revturn_60_90++; }
				}
				else if (direction_in >= 90 && direction_in < 120) {
					if (array[stepsize][3] == 0) { curve_90_120 += array[stepsize][4]; n_90_120++; }
					if (array[stepsize][3] == 0 && array[stepsize + 1][3] == 6) { revturn_90_120++; }
				}
				else if (direction_in >= 120 && direction_in < 150) {
					if (array[stepsize][3] == 0) { curve_120_150 += array[stepsize][4]; n_120_150++; }
					if (array[stepsize][3] == 0 && array[stepsize + 1][3] == 6) { revturn_120_150++; }
				}
				else if (direction_in >= 150) {
					if (array[stepsize][3] == 0) { curve_150_180 += array[stepsize][4]; n_150_180++; }
					if (array[stepsize][3] == 0 && array[stepsize + 1][3] == 6) { revturn_150_180++; }
				}
			}
#ifdef OUTPUTNEURON
			ofs << N << "," << time << "," << x << "," << y << "," << temp << "," << Direction_in << "," << state << "," << Curve << ",";
			ofs << AFD_state << "," << n.NeuronState(1) << "," << n.NeuronState(2) << "," << n.NeuronState(3) << "," << n.NeuronState(4) << "," << n.NeuronState(5)  << ",";
			ofs << AFD_output << "," << n.NeuronOutput(1) << "," << n.NeuronOutput(2) << "," << n.NeuronOutput(3) << "," << n.NeuronOutput(4) << "," << n.NeuronOutput(5) << "," << CPG_output << endl;
#endif
#ifdef OUTPUTXY
			if (time - int(time) < StepSize) {
				xy_all[int(time) - 1][2 * N - 2] = x;
				xy_all[int(time) - 1][2 * N - 1] = y;
			}
#endif
			if (time - int(time) < StepSize) {
				Index += CalculateIndex(x);
				Count++;
				if (int(time) % 60 == 0) {
					Index = Index / Count;
					ttxindex[N - 1][(int(time) % 59) - 1] = Index;
					Index = 0;
					Count = 0;
				}
			}
		}
		Index = 0;
		Count = 0;
#ifdef DISPLAYANIMAL
		cout << N << " ";
		if (N == numberofanimals) { cout << endl; }
#endif
	}
	curve_profile[0][5] = curve_0_30 / n_0_30; revturn_freq[0][5] = stepsize * 60 * revturn_0_30 / n_0_30;
	curve_profile[0][4] = curve_30_60 / n_30_60; revturn_freq[0][4] = stepsize * 60 * revturn_30_60 / n_30_60;
	curve_profile[0][3] = curve_60_90 / n_60_90; revturn_freq[0][3] = stepsize * 60 * revturn_60_90 / n_60_90;
	curve_profile[0][2] = curve_90_120 / n_90_120; revturn_freq[0][2] = stepsize * 60 * revturn_90_120 / n_90_120;
	curve_profile[0][1] = curve_120_150 / n_120_150; revturn_freq[0][1] = stepsize * 60 * revturn_120_150 / n_120_150;
	curve_profile[0][0] = curve_150_180 / n_150_180; revturn_freq[0][0] = stepsize * 60 * revturn_150_180 / n_150_180;
	
	for (int i = 0; i < 30; i++) {
		TTXindex[0][i] = 0;
		for (int N = 0; N < numberofanimals; N++) {
			TTXindex[0][i] += ttxindex[N][i];
		}
		TTXindex[0][i] = TTXindex[0][i] / numberofanimals;
	}

#ifdef SIMULATION
#ifdef OUTPUTXY
	for (int t = 0; t < 1800; t++) {
		for (int n = 0; n < 2 * numberofanimals; n++) {
			XY<< xy_all[t][n] << ",";
		}
		XY << endl;
	}
#endif
	for (int i = 0; i < 30; i++) {
		TTX << TTXindex[0][i] << endl;
#ifdef DISPLAYINDEX
		cout << setprecision(3) << TTXindex[0][i] << " ";
		if (i == 29) {	cout << endl;}
#endif
	}
	for (int i = 0; i < 6; i++) {
		ProfileCurve << curve_profile[0][i] << ",";	
	}
#ifdef OUTPUTNEURON
	ofs.close();
#endif
#ifdef OUTPUTXY
	XY.close();
#endif
	TTX.close();
	ProfileCurve.close();
#endif

	for (int i = 0; i < 30; i++) {
		IndexDiff += fabs(IndexControl[0][i] - IndexBiased[0][i]);
	}
	FitnessIndex = 0;
	for (int i = 0; i < 30; i++) {
		FitnessIndex += fabs(TTXindex[0][i] - IndexBiased[0][i]);
	}
	FitnessIndex = 1 - FitnessIndex / IndexDiff;
	if (FitnessIndex > 0) { FitnessIndex = FitnessIndex; }
	else { FitnessIndex = 0; }
	for (int i = 0; i < 6; i++) {
		CurveDiff += fabs(CurveProfile[0][i]);
	}
	FitnessCurve = 0;
	for (int i = 0; i < 6; i++) {
		FitnessCurve += fabs(curve_profile[0][i] - CurveProfile[0][i]);
	}
	FitnessCurve = 1 - FitnessCurve / CurveDiff;
	if (FitnessCurve > 0) { FitnessCurve = FitnessCurve; }
	else { FitnessCurve = 0; }

	Fitness = FitnessIndex*FitnessCurve;
#ifdef DISPLAYFITNESS
	cout << "FitnessIndex = " << FitnessIndex << " FitnessCurve = " << FitnessCurve  << " Fitness = " << Fitness << endl;
#endif

	return Fitness;
}

// ------------------------------------
// Display functions
// ------------------------------------
void EvolutionaryRunDisplay(int Generation, double BestPerf, double AvgPerf, double PerfVar)
{
#ifdef PRINTTOFILE
	cout << Generation << " " << BestPerf << " " << AvgPerf << " " << PerfVar << endl;
#else
	cout << "Generation " << Generation << ": Best = " << BestPerf << ", Average = " << AvgPerf << ", Variance = " << PerfVar << endl;
#endif
}

void ResultsDisplay(TSearch &s)
{
	TVector<double> bestVector;
	ofstream BestIndividualFile;

	bestVector = s.BestIndividual();
	BestIndividualFile.open("best.gen.dat");
	BestIndividualFile << setprecision(32);
	BestIndividualFile << bestVector << endl;
	BestIndividualFile.close();
}

// ------------------------------------
// The main program
// ------------------------------------
int main(int argc, const char* argv[]) {
	string str;
	stringstream ss;
#ifdef MIGRATIONUP
	ifstream IndexGoal("IndexUp_full.csv");
#else
	ifstream IndexGoal("IndexDown_full.csv");
#endif
	if (!IndexGoal) {
		cout << "Error: Input data file not found" << endl;
		return 1;
	}
	for (int i = 0; i < 29; i++) {
		getline(IndexGoal.seekg(0, ios_base::cur), str, ',');
		ss.str(str);
		ss >> IndexBiased[0][i];
		ss.str("");
		ss.clear(stringstream::goodbit);
	}
	getline(IndexGoal.seekg(0, ios_base::cur), str, '\n');
	ss.str(str);
	ss >> IndexBiased[0][29];
	ss.str("");
	ss.clear(stringstream::goodbit);
	IndexGoal.close();

#ifdef MIGRATIONUP
	ifstream IndexStart("IndexUp_curve-.csv");
#else
	ifstream IndexStart("IndexDown_curve-.csv");
#endif
	if (!IndexStart) {
		cout << "Error: Input data file not found" << endl;
		return 1;
	}
	for (int i = 0; i < 29; i++) {
		getline(IndexStart.seekg(0, ios_base::cur), str, ',');
		ss.str(str);
		ss >> IndexControl[0][i];
		ss.str("");
		ss.clear(stringstream::goodbit);
	}
	getline(IndexStart.seekg(0, ios_base::cur), str, '\n');
	ss.str(str);
	ss >> IndexControl[0][29];
	ss.str("");
	ss.clear(stringstream::goodbit);
	IndexStart.close();

#ifdef MIGRATIONUP
	ifstream Curve("CurveProfileUp.csv");
#else
	ifstream Curve("CurveProfileDown.csv");
#endif
	if (!Curve) {
		cout << "Error: Input data file not found" << endl;
		return 1;
	}
	for (int i = 0; i < 5; i++) {
		getline(Curve.seekg(0, ios_base::cur), str, ',');
		ss.str(str);
		ss >> CurveProfile[0][i];
		ss.str("");
		ss.clear(stringstream::goodbit);
	}
	getline(Curve.seekg(0, ios_base::cur), str, '\n');
	ss.str(str);
	ss >> CurveProfile[0][5];
	ss.str("");
	ss.clear(stringstream::goodbit);
	Curve.close();

#ifdef MIGRATIONUP
	ifstream FreqListHigh("freq_ave_20C_20C.csv");
#else
	ifstream FreqListHigh("freq_ave_20C_26C.csv");
#endif
	if (!FreqListHigh) {
		cout << "Error: Input data file not found" << endl;
		return 1;
	}
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 23; j++) {
			getline(FreqListHigh.seekg(0, ios_base::cur), str, ',');
			ss.str(str);
			ss >> FreqHigh[i][j];
			ss.str("");
			ss.clear(stringstream::goodbit);
		}
		getline(FreqListHigh.seekg(0, ios_base::cur), str, '\n');
		ss.str(str);
		ss >> FreqHigh[i][23];
		ss.str("");
		ss.clear(stringstream::goodbit);
	}
	FreqListHigh.close();

#ifdef MIGRATIONUP
	ifstream FreqListMid("freq_ave_20C_17C.csv");
#else
	ifstream FreqListMid("freq_ave_20C_23C.csv");
#endif
	if (!FreqListMid) {
		cout << "Error: Input data file not found" << endl;
		return 1;
	}
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 23; j++) {
			getline(FreqListMid.seekg(0, ios_base::cur), str, ',');
			ss.str(str);
			ss >> FreqMid[i][j];
			ss.str("");
			ss.clear(stringstream::goodbit);
		}
		getline(FreqListMid.seekg(0, ios_base::cur), str, '\n');
		ss.str(str);
		ss >> FreqMid[i][23];
		ss.str("");
		ss.clear(stringstream::goodbit);
	}
	FreqListMid.close();

#ifdef MIGRATIONUP
	ifstream FreqListLow("freq_ave_20C_14C.csv");
#else
	ifstream FreqListLow("freq_ave_20C_20C.csv");
#endif
	if (!FreqListLow) {
		cout << "Error: Input data file not found" << endl;
		return 1;
	}
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 23; j++) {
			getline(FreqListLow.seekg(0, ios_base::cur), str, ',');
			ss.str(str);
			ss >> FreqLow[i][j];
			ss.str("");
			ss.clear(stringstream::goodbit);
		}
		getline(FreqListLow.seekg(0, ios_base::cur), str, '\n');
		ss.str(str);
		ss >> FreqLow[i][23];
		ss.str("");
		ss.clear(stringstream::goodbit);
	}
	FreqListLow.close();

#ifdef MIGRATIONUP
	ifstream ProbListHigh("all_prob_ave_20C_20C.csv");
#else
	ifstream ProbListHigh("all_prob_ave_20C_26C.csv");
#endif
	if (!ProbListHigh) {
		cout << "Error: Input data file not found" << endl;
		return 1;
	}
	for (int i = 0; i < 38; i++) {
		for (int j = 0; j < 50; j++) {
			getline(ProbListHigh.seekg(0, ios_base::cur), str, ',');
			ss.str(str);
			ss >> ProbHigh[i][j];
			ss.str("");
			ss.clear(stringstream::goodbit);
		}
		getline(ProbListHigh.seekg(0, ios_base::cur), str, '\n');
		ss.str(str);
		ss >> ProbHigh[i][50];
		ss.str("");
		ss.clear(stringstream::goodbit);
	}
	ProbListHigh.close();

#ifdef MIGRATIONUP
	ifstream ProbListMid("all_prob_ave_20C_17C.csv");
#else
	ifstream ProbListMid("all_prob_ave_20C_23C.csv");
#endif
	if (!ProbListMid) {
		cout << "Error: Input data file not found" << endl;
		return 1;
	}
	for (int i = 0; i < 38; i++) {
		for (int j = 0; j < 50; j++) {
			getline(ProbListMid.seekg(0, ios_base::cur), str, ',');
			ss.str(str);
			ss >> ProbMid[i][j];
			ss.str("");
			ss.clear(stringstream::goodbit);
		}
		getline(ProbListMid.seekg(0, ios_base::cur), str, '\n');
		ss.str(str);
		ss >> ProbMid[i][50];
		ss.str("");
		ss.clear(stringstream::goodbit);
	}
	ProbListMid.close();

#ifdef MIGRATIONUP
	ifstream ProbListLow("all_prob_ave_20C_14C.csv");
#else
	ifstream ProbListLow("all_prob_ave_20C_20C.csv");
#endif
	if (!ProbListLow) {
		cout << "Error: Input data file not found" << endl;
		return 1;
	}
	for (int i = 0; i < 38; i++) {
		for (int j = 0; j < 50; j++) {
			getline(ProbListLow.seekg(0, ios_base::cur), str, ',');
			ss.str(str);
			ss >> ProbLow[i][j];
			ss.str("");
			ss.clear(stringstream::goodbit);
		}
		getline(ProbListLow.seekg(0, ios_base::cur), str, '\n');
		ss.str(str);
		ss >> ProbLow[i][50];
		ss.str("");
		ss.clear(stringstream::goodbit);
	}
	ProbListLow.close();

#ifdef MIGRATIONUP
	ifstream TimeDispersionList("time_dispersion_20C_17C.csv");
#else
	ifstream TimeDispersionList("time_dispersion_20C_23C.csv");
#endif
	if (!TimeDispersionList) {
		cout << "Error: Input data file not found" << endl;
		return 1;
	}
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 7; j++) {
			getline(TimeDispersionList.seekg(0, ios_base::cur), str, ',');
			ss.str(str);
			ss >> TimeDispersion[i][j];
			ss.str("");
			ss.clear(stringstream::goodbit);
		}
		getline(TimeDispersionList.seekg(0, ios_base::cur), str, '\n');
		ss.str(str);
		ss >> TimeDispersion[i][7];
		ss.str("");
		ss.clear(stringstream::goodbit);
	}
	TimeDispersionList.close();

	ifstream AFD("RESfunc.csv");
	//ifstream AFD("AWCRESfunc.csv");
	if (!AFD) {
		cout << "Error: Input data file not found" << endl;
		return 1;
	}
	for (int i = 0; i < 100 * stepsize; i++)
	{
		getline(AFD.seekg(0, ios_base::cur), str, '\n');
		ss.str(str);
		ss >> RESfunc[0][i];
		ss.str("");
		ss.clear(stringstream::goodbit);
	}
	AFD.close();

#ifdef SIMULATION

#ifdef PRINTTOFILE
	ofstream evolfile;
	evolfile.open("fitness.dat");
	cout.rdbuf(evolfile.rdbuf());
#endif

	RandomState rs;
	long seed = static_cast<long>(time(NULL));
	rs.SetRandomSeed(seed);

	std::cout << std::setprecision(10);
	ifstream BestIndividualFile;
	TVector<double> bestVector(1, VectSize);
	BestIndividualFile.open("best.gen.dat");
	BestIndividualFile >> bestVector;
	evaluate(bestVector, rs);

#ifdef PRINTTOFILE
	evolfile.close();
#endif

#else

	TSearch s(VectSize);
	
	// Configure the search
	long randomseed = static_cast<long>(time(NULL));
	if (argc == 2) {
		randomseed += atoi(argv[1]);
	}
	// save the seed to a file
	ofstream seedfile;
	seedfile.open("seed.dat");
	seedfile << randomseed << endl;
	seedfile.close();
	s.SetRandomSeed(randomseed);
	s.SetPopulationStatisticsDisplayFunction(EvolutionaryRunDisplay);
	s.SetSearchResultsDisplayFunction(ResultsDisplay);
	s.SetSelectionMode(RANK_BASED);
	s.SetReproductionMode(GENETIC_ALGORITHM);
	s.SetPopulationSize(PopulationSize);
	s.SetMaxGenerations(MaxGenerations);
	s.SetMutationVariance(MutationVariance);
	s.SetCrossoverProbability(0.5);
	s.SetCrossoverMode(UNIFORM);
	s.SetMaxExpectedOffspring(1.1);
	s.SetElitistFraction(0.1);
	s.SetSearchConstraint(1);
	s.SetCheckpointInterval(0);
	s.SetReEvaluationFlag(1);
	s.SetEvaluationFunction(evaluate);

#ifdef PRINTTOFILE
	ofstream evolfile;
	evolfile.open("fitness.dat");
	cout.rdbuf(evolfile.rdbuf());
#endif

	s.ExecuteSearch();

#ifdef PRINTTOFILE
	evolfile.close();
#endif
#endif

	return 0;
}