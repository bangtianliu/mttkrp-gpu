#ifndef _FLAG_H
#define _FLAG_H
#include "TTM.h"
#include <climits>
#include <cstring>
#include <limits>
using std::numeric_limits;


template<typename T>
class flag{
public:
	flag(semiTensor<T> tensor);
	T *getFlag();
	T *getBFlag();
	T *cflag;
	T *bit_flag;
	int *first;
	unsigned short *startflag;
	unsigned short *blockflag;
};


// template<typename T>
// inline T *flag<T>::getFlag(){
// 	return this->flag;
// }

// template<typename T>
// inline T *flag<T>::getBFlag(){
// 	return this->bit_flag;
// }



#endif