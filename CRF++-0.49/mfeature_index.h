#ifndef _MFEATURE_INDEX_H_
#define _MFEATURE_INDEX_H_

#include "feature_index.h"
#include "node.h"

namespace CRFPP {

class MTaggerImpl;
class MEncoderFeatureIndex: public FeatureIndex {
private:
	int getID(const char *) {
		return -1;
	}
public:
    explicit MEncoderFeatureIndex(size_t n) {
      thread_num_ = n;
	  size_ = 0;
	  max_tagger_len = 0;
      init();
	  node_.resize(thread_num_);
    }
	~MEncoderFeatureIndex() {
		for (std::vector<Node **>::iterator it = node_.begin();
			it != node_.end(); ++it) 
				delete *it;
	}
	virtual bool open(const char*, const char*);
	virtual void clear() { }
	unsigned int size() { 
		return size_;
	}
	unsigned short ysize() { return ysize_; }
	void set_ysize(unsigned short y ) { 
		y_.resize(y);
		ysize_ = y; 
	}
	void rebuildFeatures(MTaggerImpl *);
	void set_max_tagger(int n);
protected:
	Mmap <char> mmap_;
	unsigned int size_;
	unsigned short ysize_;
	unsigned int max_tagger_len;
	//scoped_array< FreeList<Path> > node;
	std::vector <Node **> node_; 
};

}

#endif