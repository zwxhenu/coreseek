#include <iostream>
#include <vector>
#include <queue>
#include "param.h"
#include "crfpp.h"
#include "scoped_ptr.h"
#include "feature_index.h"
#include "tagger.h"
#include "mtagger.h"

namespace CRFPP {

bool MTaggerImpl::open(MEncoderFeatureIndex *f) {
	mode_ = LEARN;
	feature_index_ = f;
	ysize_ = feature_index_->ysize();
	return true;
}

void MTaggerImpl::import(char *p)
{
	char *ptr = p;
	read_static<unsigned int>(&ptr, &x_size);
	answer_ = (unsigned short int*)ptr;
	ptr += sizeof(unsigned short int)*x_size;
	unigram_features_idx = (unsigned short int*)ptr;
	ptr += sizeof(unsigned short int)*(x_size*2-1);
	feature_base_ = ptr;
	result_.resize(x_size);
	//node_.resize(x_size);
	//TODO how to deal node_?
	//node_.resize(x_size*ysize_); //x*y
	node_ =NULL;
}

int * MTaggerImpl::feature(int p) {
	if(!p)
		return (int*)(feature_base_);
	int idx = p - 1;
	if(idx < (int)x_size){
		//uni-gram
		return (int*)(feature_base_ + unigram_features_idx[idx]*sizeof(int));
	}else{
		//bi-gram
		return (int*)(feature_base_ + 
			(unigram_features_idx[idx]+unigram_features_idx[x_size-1])*sizeof(int));
	}
	return NULL;
}
void MTaggerImpl::buildLattice() {
	if (!x_size) return;

	feature_index_->rebuildFeatures(this);

	for (size_t i = 0; i < x_size; ++i) {
		for (size_t j = 0; j < ysize_; ++j) {
			feature_index_->calcCost(node_[i*ysize_ + j]);
			const std::vector<Path *> &lpath = node_[i*ysize_ + j]->lpath;
			for (const_Path_iterator it = lpath.begin(); it != lpath.end(); ++it)
				feature_index_->calcCost(*it);
		}
	}
}

void MTaggerImpl::forwardbackward() {
	if (!x_size) return;

	for (int i = 0; i < static_cast<int>(x_size); ++i)
		for (size_t j = 0; j < ysize_; ++j)
			node_[i*ysize_ + j]->calcAlpha();

	for (int i = static_cast<int>(x_size - 1); i >= 0;  --i)
		for (size_t j = 0; j < ysize_; ++j)
			node_[i*ysize_ + j]->calcBeta();

	Z_ = 0.0;
	for (size_t j = 0; j < ysize_; ++j)
		Z_ = logsumexp(Z_, node_[0*ysize_ + j]->beta, j == 0);

	return;
}

double MTaggerImpl::gradient(double *expected) {
	if (!x_size) return 0.0;

	buildLattice();
	forwardbackward();
	double s = 0.0;

	for (size_t i = 0;   i < x_size; ++i)
		for (size_t j = 0; j < ysize_; ++j)
			node_[i*ysize_ + j]->calcExpectation(expected, Z_, ysize_);

	for (size_t i = 0;   i < x_size; ++i) {
		for (int *f = node_[i*ysize_ + answer_[i]]->fvector; *f != -1; ++f)
			--expected[*f + answer_[i]];
		s += node_[i*ysize_ + answer_[i]]->cost;  // UNIGRAM cost
		const std::vector<Path *> &lpath = node_[i*ysize_ + answer_[i]]->lpath;
		for (const_Path_iterator it = lpath.begin(); it != lpath.end(); ++it) {
			if ((*it)->lnode->y == answer_[(*it)->lnode->x]) {
				for (int *f = (*it)->fvector; *f != -1; ++f)
					--expected[*f +(*it)->lnode->y * ysize_ +(*it)->rnode->y];
				s += (*it)->cost;  // BIGRAM COST
				break;
			}
		}
	}

	viterbi();  // call for eval()

	return Z_ - s ;
}

void MTaggerImpl::viterbi() {
	for (size_t i = 0;   i < x_size; ++i) {
		for (size_t j = 0; j < ysize_; ++j) {
			double bestc = -1e37;
			Node *best = 0;
			const std::vector<Path *> &lpath = node_[i*ysize_ + j]->lpath;
			for (const_Path_iterator it = lpath.begin(); it != lpath.end(); ++it) {
				double cost = (*it)->lnode->bestCost +(*it)->cost +
					node_[i*ysize_ +j]->cost;
				if (cost > bestc) {
					bestc = cost;
					best  = (*it)->lnode;
				}
			}
			node_[i*ysize_ +j]->prev     = best;
			node_[i*ysize_ +j]->bestCost = best ? bestc : node_[i*ysize_ +j]->cost;
		}
	}

	double bestc = -1e37;
	Node *best = 0;
	size_t s = x_size -1;
	for (size_t j = 0; j < ysize_; ++j) {
		if (bestc < node_[s*ysize_ +j]->bestCost) {
			best  = node_[s*ysize_ +j];
			bestc = node_[s*ysize_ +j]->bestCost;
		}
	}

	for (Node *n = best; n; n = n->prev)
		result_[n->x] = n->y;

	cost_ = -node_[(x_size-1)*ysize_ + result_[x_size-1]]->bestCost;
}

int MTaggerImpl::eval() {
	int err = 0;
	for (size_t i = 0; i < x_size; ++i)
		if (answer_[i] != result_[i]) { ++err; }
		return err;
}

} //end name space