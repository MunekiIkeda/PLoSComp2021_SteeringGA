main: main.o NervousSystem.o TSearch.o random.o
	g++ -pthread -o main main.o NervousSystem.o TSearch.o random.o
random.o: random.cpp random.h VectorMatrix.h
	g++ -c -O3 -flto random.cpp
TSearch.o: TSearch.cpp TSearch.h
	g++ -c -O3 -flto TSearch.cpp
NervousSystem.o: NervousSystem.cpp NervousSystem.h random.h
	g++ -c -O3 -flto NervousSystem.cpp
CurveSearch.o: main.cpp NervousSystem.h TSearch.h random.h VectorMatrix.h
	g++ -c -O3 -flto main.cpp
clean:
	rm *.o main
