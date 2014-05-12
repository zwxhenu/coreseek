#include "feature_index.h"
#include "mfeature_index.h"
#include "mtagger.h"

namespace CRFPP {

bool MEncoderFeatureIndex::open(const char* filename1, const char*)
{
	CHECK_FALSE(mmap_.open(filename1)) << mmap_.what();

	char *ptr = mmap_.begin();

	read_static<unsigned int>(&ptr, &size_); //we do not care about other things yet. for feature-id is pre-computed in binary train file.
	
	mmap_.close();	
	return true;
}
void MEncoderFeatureIndex::set_max_tagger(int n)
{
	for (std::vector<Node **>::iterator it = node_.begin();
		it != node_.end(); ++it) {
		//delete *it;
		*it = new Node*[(n+1)*(ysize_+1)]; //make the buffer a bit larger.
	}
}

void MEncoderFeatureIndex::rebuildFeatures(MTaggerImpl *tagger) {
	size_t fid = 0;
	unsigned short thread_id = tagger->thread_id();

	path_freelist_[thread_id].free();
	node_freelist_[thread_id].free();
	tagger->set_node_buffer(node_[thread_id]);

	for (size_t cur = 0; cur < tagger->size(); ++cur) {
		int *f = tagger->feature(fid++);
		for (size_t i = 0; i < ysize_; ++i) {
			Node *n = node_freelist_[thread_id].alloc();
			n->clear();
			n->x = cur;
			n->y = i;
			n->fvector = f;
			tagger->set_node(n, cur, i);
		}
	}

	for (size_t cur = 1; cur < tagger->size(); ++cur) {
		int *f  = tagger->feature(fid++);
		for (size_t j = 0; j < ysize_; ++j) {
			for (size_t i = 0; i < ysize_; ++i) {
				Path *p = path_freelist_[thread_id].alloc();
				p->clear();
				p->add(tagger->node(cur-1, j),
					tagger->node(cur, i));
				p->fvector = f;
			}
		}
	}
}

}