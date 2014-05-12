#ifndef _MTAGGER_H_
#define _MTAGGER_H_

#include <iostream>
#include <vector>
#include <queue>
#include "param.h"
#include "crfpp.h"
#include "scoped_ptr.h"
#include "feature_index.h"
#include "tagger.h"
#include "mfeature_index.h"

namespace CRFPP {
 
class MTaggerImpl {
  private:

    struct QueueElement {
      Node *node;
      QueueElement *next;
      double fx;
      double gx;
    };

    class QueueElementComp {
    public:
      const bool operator()(QueueElement *q1,
                            QueueElement *q2)
        { return(q1->fx > q2->fx); }
    };
	
	unsigned short                    thread_id_;
    MEncoderFeatureIndex                     *feature_index_;
    //std::vector <std::vector <Node *> > node_;
    //std::vector <Node *> node_; 
	Node ** node_; 
	std::vector <unsigned short int>  result_;
    whatlog what_;
    string_buffer os_;

    scoped_ptr<std::priority_queue <QueueElement*, std::vector <QueueElement *>,
      QueueElementComp> > agenda_;
    scoped_ptr<FreeList <QueueElement> > nbest_freelist_;

	enum { TEST, LEARN };
	unsigned int mode_   : 2;

	double                cost_;
	double                Z_;
	unsigned short        ysize_;

protected:
	void forwardbackward();
	void viterbi();
	void buildLattice();
	bool initNbest();

public:
	explicit MTaggerImpl():thread_id_(0), x_size(0), Z_(0), ysize_(0),cost_(0),
				feature_index_(NULL) {}
    virtual ~MTaggerImpl() { }
	void import(char*);
	bool open(MEncoderFeatureIndex *f);

	void   set_thread_id(unsigned short id) { thread_id_ = id; }
	unsigned short thread_id() { return thread_id_; }
	unsigned int size() { return x_size; }
	double cost() const { return cost_; }
	double Z() const { return Z_; }
	void set_node_buffer(Node ** n) {	node_ = n;	};

	int          eval();
	double       gradient(double *);
	int * feature(int);

	Node  *node(size_t i, size_t j) { return node_[i*ysize_ + j]; }
	void   set_node(Node *n, size_t i, size_t j) { node_[i*ysize_ + j] = n; }

    const char* what() { return what_.str(); }
protected:
	unsigned int x_size;
	unsigned short int* answer_;
	unsigned short int* unigram_features_idx;
	char*				feature_base_;
};

} //end name space

#endif