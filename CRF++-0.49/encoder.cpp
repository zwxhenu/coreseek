//
//  CRF++ -- Yet Another CRF toolkit
//
//  $Id: encoder.cpp 1601 2007-03-31 09:47:18Z taku $;
//
//  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
//
#include <fstream>
#include "param.h"
#include "encoder.h"
#include "timer.h"
#include "tagger.h"
#include "lbfgs.h"
#include "common.h"
#include "feature_index.h"
#include "scoped_ptr.h"
#include "thread.h"
#include "mfeature_index.h"
#include "mtagger.h"

namespace {
  bool toLower(std::string *s) {
    for (size_t i = 0; i < s->size(); ++i) {
      char c = (*s)[i];
      if ((c >= 'A') && (c <= 'Z')) {
        c += 'a' - 'A';
        (*s)[i] = c;
      }
    }
    return true;
  }
}

namespace CRFPP {

  class CRFEncoderThread: public thread {
  public:
    TaggerImpl **x;
    unsigned short start_i;
    unsigned short thread_num;
    int zeroone;
    int err;
    size_t size;
    double obj;
    std::vector<double> expected;

    void run() {
      obj = 0.0;
      err = zeroone = 0;
      std::fill(expected.begin(), expected.end(), 0.0);
      for (size_t i = start_i; i < size; i += thread_num) {
        obj += x[i]->gradient(&expected[0]);
        int error_num = x[i]->eval();
        err += error_num;
        if (error_num) ++zeroone;
      }
    }
  };

  class CRFMEncoderThread: public thread {
  public:
	  MTaggerImpl *x;
	  unsigned short start_i;
	  unsigned short thread_num;
	  int zeroone;
	  int err;
	  size_t size;
	  double obj;
	  std::vector<double> expected;

	  void run() {
		  obj = 0.0;
		  err = zeroone = 0;
		  std::fill(expected.begin(), expected.end(), 0.0);
		  for (size_t i = start_i; i < size; i += thread_num) {
			  obj += x[i].gradient(&expected[0]);
			  int error_num = x[i].eval();
			  err += error_num;
			  if (error_num) ++zeroone;
		  }
	  }
  };

  bool runMIRA(const std::vector<TaggerImpl* > &x,
               EncoderFeatureIndex *feature_index,
               double *alpha,
               size_t maxitr,
               float C,
               double eta,
               unsigned short shrinking_size,
               unsigned short thread_num) {
    std::vector<unsigned char> shrink(x.size());
    std::vector<float> upper_bound(x.size());
    std::vector<double> expected(feature_index->size());

    std::fill(upper_bound.begin(), upper_bound.end(), 0.0);
    std::fill(shrink.begin(), shrink.end(), 0);

    int converge = 0;
    int all = 0;
    for (size_t i = 0; i < x.size(); ++i)  all += (int)x[i]->size();

    for (size_t itr = 0; itr < maxitr; ++itr) {
      int zeroone = 0;
      int err = 0;
      int active_set = 0;
      int upper_active_set = 0;
      double max_kkt_violation = 0.0;

      feature_index->clear();

      for (size_t i = 0; i < x.size(); ++i) {
        if (shrink[i] >= shrinking_size) continue;

        ++active_set;
        std::fill(expected.begin(), expected.end(), 0.0);
        double cost_diff = x[i]->collins(&expected[0]);
        int error_num = x[i]->eval();
        err += error_num;
        if (error_num) ++zeroone;

        if (error_num == 0) {
          ++shrink[i];
        } else {
          shrink[i] = 0;
          double s = 0.0;
          for (size_t k = 0; k < expected.size(); ++k)
            s += expected[k] * expected[k];

          double mu = _max(0.0, (error_num - cost_diff) / s);

          if (upper_bound[i] + mu > C) {
            mu = C - upper_bound[i];
            ++upper_active_set;
          } else {
            max_kkt_violation = _max(error_num - cost_diff,
                                     max_kkt_violation);
          }

          if (mu > 1e-10) {
            upper_bound[i] += mu;
            upper_bound[i] = _min(C, upper_bound[i]);
            for (size_t k = 0; k < expected.size(); ++k)
              alpha[k] += mu * expected[k];
          }
        }
      }

      double obj = 0.0;
      for (size_t i = 0; i < feature_index->size(); ++i)
        obj += alpha[i] * alpha[i];

      std::cout << "iter="  << itr
                << " terr=" << 1.0 * err / all
                << " serr=" << 1.0 * zeroone / x.size()
                << " act=" <<  active_set
                << " uact=" << upper_active_set
                << " obj=" << obj
                << " kkt=" << max_kkt_violation << std::endl;

      if (max_kkt_violation <= 0.0) {
        std::fill(shrink.begin(), shrink.end(), 0);
        converge++;
      } else {
        converge = 0;
      }

      if (itr > maxitr || converge == 2)  break;  // 2 is ad-hoc
    }

    return true;
  }

  bool runCRF(const std::vector<TaggerImpl* > &x,
              EncoderFeatureIndex *feature_index,
              double *alpha,
              size_t maxitr,
              float C,
              double eta,
              unsigned short shrinking_size,
              unsigned short thread_num,
              bool orthant) {
    double old_obj = 1e+37;
    int    converge = 0;
    LBFGS lbfgs;
    std::vector<CRFEncoderThread> thread(thread_num);

    for (size_t i = 0; i < thread_num; i++) {
      thread[i].start_i = i;
      thread[i].size = x.size();
      thread[i].thread_num = thread_num;
      thread[i].x = const_cast<TaggerImpl **>(&x[0]);
      thread[i].expected.resize(feature_index->size());
    }

    size_t all = 0;
    for (size_t i = 0; i < x.size(); ++i)  all += x[i]->size();

    for (size_t itr = 0; itr < maxitr; ++itr) {
      feature_index->clear();

      for (size_t i = 0; i < thread_num; ++i) thread[i].start();
      for (size_t i = 0; i < thread_num; ++i) thread[i].join();

      for (size_t i = 1; i < thread_num; ++i) {
        thread[0].obj += thread[i].obj;
        thread[0].err += thread[i].err;
        thread[0].zeroone += thread[i].zeroone;
      }

      for (size_t i = 1; i < thread_num; ++i) {
        for (size_t k = 0; k < feature_index->size(); ++k)
          thread[0].expected[k] += thread[i].expected[k];
      }

      size_t num_nonzero = 0;
      if (orthant) {   // L1
        for (size_t k = 0; k < feature_index->size(); ++k) {
          thread[0].obj += std::abs(alpha[k] / C);
          if (alpha[k] != 0.0) ++num_nonzero;
        }
      } else {
        num_nonzero = feature_index->size();
        for (size_t k = 0; k < feature_index->size(); ++k) {
          thread[0].obj += (alpha[k] * alpha[k] /(2.0 * C));
          thread[0].expected[k] += alpha[k] / C;
        }
      }

      double diff = (itr == 0 ? 1.0 :
                     std::abs(old_obj - thread[0].obj)/old_obj);
      std::cout << "iter="  << itr
                << " terr=" << 1.0 * thread[0].err / all
                << " serr=" << 1.0 * thread[0].zeroone / x.size()
                << " act=" << num_nonzero
                << " obj=" << thread[0].obj
                << " diff="  << diff << std::endl;
      old_obj = thread[0].obj;

      if (diff < eta)
        converge++;
      else
        converge = 0;

      if (itr > maxitr || converge == 3)  break;  // 3 is ad-hoc

      if (lbfgs.optimize(feature_index->size(),
                         &alpha[0],
                         thread[0].obj,
                         &thread[0].expected[0], orthant, C) <= 0)
        return false;
    }

    return true;
  }


bool runCRF(MTaggerImpl* x, unsigned int x_len, 
	MEncoderFeatureIndex *feature_index,
	double *alpha,
	size_t maxitr,
	float C,
	double eta,
	unsigned short shrinking_size,
	unsigned short thread_num,
	bool orthant) {
		double old_obj = 1e+37;
		int    converge = 0;
		LBFGS lbfgs;
		std::vector<CRFMEncoderThread> thread(thread_num);

		for (size_t i = 0; i < thread_num; i++) {
			thread[i].start_i = i;
			thread[i].size = x_len;
			thread[i].thread_num = thread_num;
			thread[i].x = const_cast<MTaggerImpl *>(&x[0]);
			thread[i].expected.resize(feature_index->size());
		}

		size_t all = 0;
		for (size_t i = 0; i < x_len; ++i)  all += x[i].size();

		for (size_t itr = 0; itr < maxitr; ++itr) {
			feature_index->clear();

			for (size_t i = 0; i < thread_num; ++i) thread[i].start();
			for (size_t i = 0; i < thread_num; ++i) thread[i].join();

			for (size_t i = 1; i < thread_num; ++i) {
				thread[0].obj += thread[i].obj;
				thread[0].err += thread[i].err;
				thread[0].zeroone += thread[i].zeroone;
			}

			for (size_t i = 1; i < thread_num; ++i) {
				for (size_t k = 0; k < feature_index->size(); ++k)
					thread[0].expected[k] += thread[i].expected[k];
			}

			size_t num_nonzero = 0;
			if (orthant) {   // L1
				for (size_t k = 0; k < feature_index->size(); ++k) {
					thread[0].obj += std::abs(alpha[k] / C);
					if (alpha[k] != 0.0) ++num_nonzero;
				}
			} else {
				num_nonzero = feature_index->size();
				for (size_t k = 0; k < feature_index->size(); ++k) {
					thread[0].obj += (alpha[k] * alpha[k] /(2.0 * C));
					thread[0].expected[k] += alpha[k] / C;
				}
			}

			double diff = (itr == 0 ? 1.0 :
			std::abs(old_obj - thread[0].obj)/old_obj);
			std::cout << "iter="  << itr
				<< " terr=" << 1.0 * thread[0].err / all
				<< " serr=" << 1.0 * thread[0].zeroone / x_len
				<< " act=" << num_nonzero
				<< " obj=" << thread[0].obj
				<< " diff="  << diff << std::endl;
			old_obj = thread[0].obj;

			if (diff < eta)
				converge++;
			else
				converge = 0;

			if (itr > maxitr || converge == 3)  break;  // 3 is ad-hoc

			if (lbfgs.optimize(feature_index->size(),
				&alpha[0],
				thread[0].obj,
				&thread[0].expected[0], orthant, C) <= 0)
				return false;
		}

		return true;
	}
  bool Encoder::convert(const char* textfilename,
                        const char *binaryfilename) {
    EncoderFeatureIndex feature_index(1);
    CHECK_FALSE(feature_index.convert(textfilename, binaryfilename))
      << feature_index.what();

    return true;
  }
  bool ExportToMmap(std::vector<TaggerImpl* >& x, 
		EncoderFeatureIndex& feature_index,
		const char *modelfile)
  {
	  //do save features_index
	  char sufix_feature[] = ".features";
	  char sufix_train[] = ".train";
	  char buf[512];

	  size_t model_filename_len = strlen(modelfile);
	  memcpy(buf,modelfile, model_filename_len);
	  memcpy(&buf[model_filename_len],sufix_feature,strlen(sufix_feature));
	  buf[model_filename_len + strlen(sufix_feature) ] = 0;
	  feature_index.export_mmap(buf);
	  //write corpus
	  memcpy(&buf[model_filename_len],sufix_train,strlen(sufix_train));
	  buf[model_filename_len + strlen(sufix_train) ] = 0;
	  //mmap-train file format
	  //size, max-tagger-len, tag_count, tag-list.  
	  //		tagger-list(id->offset), tagger-data
	  //tagger-data, len: answer_[len], node_feature_list[len]
	  //after current item-data is bigram-feature-list
	  /*
	  size_t ysize() const { return y_.size(); }
	  const char* y(size_t i) const { return y_[i]; }
	  if (mode_ == LEARN) {
	  size_t r = ysize_;
	  for (size_t k = 0; k < ysize_; ++k)
	  if (std::strcmp(yname(k), column[xsize]) == 0)
	  r = k;

	  CHECK_FALSE(r != ysize_) << "cannot find answer: " << column[xsize];
	  answer_[s] = r;
	  }

	  */
	  TaggerImpl* max_len_tagger = NULL;
	  unsigned int x_len = x.size();
	  unsigned int* tagger_offset = new unsigned int[x_len];
	  unsigned int offset = 0;
	  for (size_t i = 0; i<x_len; i++ ) {
		size_t x_size = x[i]->size();
        if(!max_len_tagger || x_size > max_len_tagger->size())
			max_len_tagger = x[i];

		//calculate offset
		tagger_offset[i] = offset;
		offset += sizeof(unsigned int); //the lengh of current tagger.
		offset += sizeof(unsigned short int)*x_size; //answer_[i]
		offset += sizeof(unsigned short int)*(x_size*2-1); // feature-index.

		size_t fid = x[i]->feature_id();
		for (size_t cur = 0; cur < x_size; ++cur) {
			int *f = feature_index.features(fid++);
			int *fbegin = f;
			while(*f != -1) 
				f++;				
			offset += (f-fbegin + 1)*sizeof(int); //the uni-gram feature-list	
		}
		for (size_t cur = 1; cur < x_size; ++cur) {
			int *f = feature_index.features(fid++);
			int *fbegin = f;
			while(*f != -1) 
				f++;				
			offset += (f-fbegin + 1)*sizeof(int); //the bi-gram feature-list
		}
	  }//end for
	  //begin write file
	  {
		  std::ofstream bofs;
		  bofs.open(buf, OUTPUT_MODE);
		  bofs.write(reinterpret_cast<char *>(&x_len), sizeof(x_len));
		  unsigned int max_len = max_len_tagger->size();
		  bofs.write(reinterpret_cast<char *>(&max_len), sizeof(max_len));
		  //write pos-tag
		  unsigned short key_id = feature_index.ysize();
		  bofs.write(reinterpret_cast<char *>(&key_id), sizeof(key_id));
		  key_id = 0;
		  unsigned short str_len = 0;
		  for(size_t i = 0; i< feature_index.ysize(); i++ ) {
			  //format id, keylen, key.
			  const char* str = feature_index.y(i);
			  key_id = i;
			  str_len = strlen(str);
			  bofs.write(reinterpret_cast<char *>(&key_id), sizeof(key_id));
			  bofs.write(reinterpret_cast<char *>(&str_len), sizeof(str_len));
			  if(str_len) {
				  bofs.write(str , str_len);
			  }
		  }
		  //write offset-index
		  bofs.write(reinterpret_cast<char *>(&tagger_offset[0]), x_len * sizeof(unsigned int));
		  //write each tagger
		  offset = 0;
		  for (size_t i = 0; i<x_len; i++ ) {
			  unsigned int x_size = x[i]->size();
			  std::vector <unsigned short int>& answer = x[i]->answer();
			  //calculate offset
			  bofs.write(reinterpret_cast<char *>(&x_size), sizeof(x_size));
			  offset += sizeof(unsigned int); //the length of current tagger.
			  bofs.write(reinterpret_cast<char *>(&answer[0]), 
					sizeof(unsigned short int)*x_size);
			  //to do should check answer.size() == x_size
			  offset += sizeof(unsigned short int)*x_size; //answer_[i]

			  unsigned short int* features = new unsigned short int[x_size*2-1];
			  offset += sizeof(unsigned short int)*(x_size*2-1); // feature-count.

			  size_t fid = x[i]->feature_id();
			  for (size_t cur = 0; cur < x_size; ++cur) {
				  int *f = feature_index.features(fid++);
				  int *fbegin = f;
				  while(*f != -1) 
					  f++;	
				  features[cur] = f-fbegin + 1;
				  if(cur)
					  features[cur] += features[cur-1]; //add it up
			  }
			  for (size_t cur = 1; cur < x_size; ++cur) {
				  int *f = feature_index.features(fid++);
				  int *fbegin = f;
				  while(*f != -1) 
					  f++;
				  size_t pos = cur + x_size - 1;
				  features[pos] = f-fbegin + 1;
				  if(cur>1)
					  features[pos] += features[pos-1]; //add it up
			  }

			  bofs.write(reinterpret_cast<char *>(&features[0]), 
				  sizeof(unsigned short int)*(x_size*2-1)); //FIXME: This limited total feature in the tagger can not more than 65536
			  
			  fid = x[i]->feature_id(); //reset fid
			  for (size_t cur = 0; cur < x_size; ++cur) {
				  int *f = feature_index.features(fid++);
				  int *fbegin = f;
				  while(*f != -1) {
					  bofs.write(reinterpret_cast<char *>(&f[0]), sizeof(int));
					  f++;	
				  }
				  //write the end
				  int fend = -1;
				  bofs.write(reinterpret_cast<char *>(&fend), sizeof(fend));
			  }
			  for (size_t cur = 1; cur < x_size; ++cur) {
				  int *f = feature_index.features(fid++);
				  int *fbegin = f;
				  while(*f != -1) {
					  bofs.write(reinterpret_cast<char *>(f), sizeof(int));
					  f++;	
				  }
				  //write the end
				  int fend = -1;
				  bofs.write(reinterpret_cast<char *>(&fend), sizeof(fend));
			  }
		  }//end for
		  bofs.close();
	  }
	  delete[] tagger_offset;
	  return 0;
  }

  bool Encoder::train(const char *templfile,
			const char *modelfile,
			bool textmodelfile,
			size_t maxitr,
			size_t freq,
			double eta,
			double C,
			unsigned short thread_num,
			unsigned short shrinking_size,
			int algorithm) {
		std::cout << COPYRIGHT << std::endl;

		CHECK_FALSE(eta > 0.0) << "eta must be > 0.0";
		CHECK_FALSE(C >= 0.0) << "C must be >= 0.0";
		CHECK_FALSE(shrinking_size >= 1) << "shrinking-size must be >= 1";
		CHECK_FALSE(thread_num > 0) << "thread must be > 0";

#ifndef CRFPP_USE_THREAD
		CHECK_FALSE(thread_num == 1)
			<< "This architecture doesn't support multi-thrading";
#endif

		CHECK_FALSE(algorithm == CRF_L2 || algorithm == CRF_L1 
			|| algorithm == MMAP ||
			(algorithm == MIRA && thread_num == 1))
			<<  "MIRA doesn't support multi-thrading";

#define WHAT_ERROR(msg) do { \
	delete[] x; \
	std::cerr << msg << std::endl; \
	return false; } while (0)

		MEncoderFeatureIndex feature_index(thread_num);
		MTaggerImpl* x = NULL;
		size_t s = sizeof(MTaggerImpl);

		std::cout.setf(std::ios::fixed, std::ios::floatfield);
		std::cout.precision(5);
		//load features

		char sufix_feature[] = ".features";
		char sufix_train[] = ".train";
		char buf[512];

		size_t model_filename_len = strlen(modelfile);
		memcpy(buf,modelfile, model_filename_len);
		memcpy(&buf[model_filename_len],sufix_feature,strlen(sufix_feature));
		buf[model_filename_len + strlen(sufix_feature) ] = 0;
		feature_index.open(buf,NULL);

		std::vector <double> alpha(feature_index.size());           // parameter
		std::fill(alpha.begin(), alpha.end(), 0.0);
		feature_index.set_alpha(&alpha[0]);
		
		//load pos-tagger
		Mmap <char> mmap_;
		memcpy(&buf[model_filename_len],sufix_train,strlen(sufix_train));
		buf[model_filename_len + strlen(sufix_train) ] = 0;

		CHECK_FALSE(mmap_.open(buf)) << mmap_.what();
		unsigned int x_len = 0;
		unsigned int max_len = 0;
		{
			unsigned short pos_len = 0;
			char *ptr = mmap_.begin();
			read_static<unsigned int>(&ptr, &x_len);
			read_static<unsigned int>(&ptr, &max_len);
			read_static<unsigned short>(&ptr, &pos_len); //the y-size()
			feature_index.set_ysize(pos_len);
			feature_index.set_max_tagger(max_len);
			x = new MTaggerImpl[x_len];
			//skip pos-list
			for(size_t i = 0; i<pos_len; i++) {
				unsigned short pos_id;
				unsigned short pos_str_len;
				read_static<unsigned short>(&ptr, &pos_id);
				read_static<unsigned short>(&ptr, &pos_str_len);
				ptr += pos_str_len; //skip the string data.
			}
			unsigned int* tagger_offset = (unsigned int*)(ptr);
			ptr += x_len * sizeof(unsigned int);
			//skip index
			for(size_t i = 0;i < x_len; i++) {
				char* cur = ptr+tagger_offset[i];
				x[i].open(&feature_index);
				x[i].import(cur);
				x[i].set_thread_id(i % thread_num);
			}
		}
			
		std::cout << "Number of sentences: " << x_len << std::endl;
		std::cout << "Number of features:  " << feature_index.size() << std::endl;
		std::cout << "Number of thread(s): " << thread_num << std::endl;
		std::cout << "Freq:                " << freq << std::endl;
		std::cout << "eta:                 " << eta << std::endl;
		std::cout << "C:                   " << C << std::endl;
		std::cout << "shrinking size:      " << shrinking_size<< std::endl;
		//begin estimator
		progress_timer pg;

		switch (algorithm) {
		case CRF_L2:
			if (!runCRF(x, x_len, &feature_index, &alpha[0],
				maxitr, C, eta, shrinking_size, thread_num, false))
				WHAT_ERROR("CRF_L2 execute error");
			break;
		case CRF_L1:
			if (!runCRF(x, x_len, &feature_index, &alpha[0],
				maxitr, C, eta, shrinking_size, thread_num, true))
				WHAT_ERROR("CRF_L1 execute error");
			break;
		}
		//clear 
		if(x)
			delete[] x;
		//save module
		mmap_.close();
		return true;
  }
  bool Encoder::learn(const char *templfile,
                      const char *trainfile,
                      const char *modelfile,
                      bool textmodelfile,
                      size_t maxitr,
                      size_t freq,
                      double eta,
                      double C,
                      unsigned short thread_num,
                      unsigned short shrinking_size,
                      int algorithm) {
    std::cout << COPYRIGHT << std::endl;

    CHECK_FALSE(eta > 0.0) << "eta must be > 0.0";
    CHECK_FALSE(C >= 0.0) << "C must be >= 0.0";
    CHECK_FALSE(shrinking_size >= 1) << "shrinking-size must be >= 1";
    CHECK_FALSE(thread_num > 0) << "thread must be > 0";

#ifndef CRFPP_USE_THREAD
    CHECK_FALSE(thread_num == 1)
      << "This architecture doesn't support multi-thrading";
#endif

    CHECK_FALSE(algorithm == CRF_L2 || algorithm == CRF_L1 
				|| algorithm == MMAP ||
                (algorithm == MIRA && thread_num == 1))
                  <<  "MIRA doesn't support multi-thrading";

    EncoderFeatureIndex feature_index(thread_num);
    std::vector<TaggerImpl* > x;

    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(5);
#undef WHAT_ERROR
#define WHAT_ERROR(msg) do { \
    for (std::vector<TaggerImpl *>::iterator it = x.begin(); \
         it != x.end(); ++it) \
      delete *it; \
    std::cerr << msg << std::endl; \
    return false; } while (0)

    CHECK_FALSE(feature_index.open(templfile, trainfile))
      << feature_index.what();

    {
      progress_timer pg;

      std::ifstream ifs(trainfile);
      CHECK_FALSE(ifs) << "cannot open: " << trainfile;

      std::cout << "reading training data: " << std::flush;
      size_t line = 0;
      while (ifs) {
        TaggerImpl *_x = new TaggerImpl();
        _x->open(&feature_index);
        _x->set_thread_id(line % thread_num);
        if (!_x->read(&ifs) || !_x->shrink())
          WHAT_ERROR(_x->what());

        if (!_x->empty())
          x.push_back(_x);
        else
          delete _x;

        if (++line % 100 == 0) std::cout << line << ".. " << std::flush;
      }

      ifs.close();
      std::cout << "\nDone!";
    }

    feature_index.shrink(freq);

    std::vector <double> alpha(feature_index.size());           // parameter
    std::fill(alpha.begin(), alpha.end(), 0.0);
    feature_index.set_alpha(&alpha[0]);

    std::cout << "Number of sentences: " << x.size() << std::endl;
    std::cout << "Number of features:  " << feature_index.size() << std::endl;
    std::cout << "Number of thread(s): " << thread_num << std::endl;
    std::cout << "Freq:                " << freq << std::endl;
    std::cout << "eta:                 " << eta << std::endl;
    std::cout << "C:                   " << C << std::endl;
    std::cout << "shrinking size:      " << shrinking_size
              << std::endl;

    progress_timer pg;

    switch (algorithm) {
    case MIRA:
      if (!runMIRA(x, &feature_index, &alpha[0],
                   maxitr, C, eta, shrinking_size, thread_num))
        WHAT_ERROR("MIRA execute error");
      break;
    case CRF_L2:
      if (!runCRF(x, &feature_index, &alpha[0],
                  maxitr, C, eta, shrinking_size, thread_num, false))
        WHAT_ERROR("CRF_L2 execute error");
      break;
    case CRF_L1:
      if (!runCRF(x, &feature_index, &alpha[0],
                  maxitr, C, eta, shrinking_size, thread_num, true))
        WHAT_ERROR("CRF_L1 execute error");
      break;
	case MMAP:
	  //save feature-list & corpus file to mem-mapping file for later crf-usage
	  ExportToMmap(x,feature_index,modelfile);
	  break;
    }

    for (std::vector<TaggerImpl *>::iterator it = x.begin();
         it != x.end(); ++it)
      delete *it;
	
	if(algorithm != MMAP) {
		if (!feature_index.save(modelfile, textmodelfile))
			WHAT_ERROR(feature_index.what());
	}
    std::cout << "\nDone!";

    return true;
  }
}

int crfpp_learn(int argc, char **argv) {
  static const CRFPP::Option long_options[] = {
    {"freq",     'f', "1",      "INT",
     "use features that occuer no less than INT(default 1)" },
    {"maxiter" , 'm', "100000", "INT",
     "set INT for max iterations in LBFGS routine(default 10k)" },
    {"cost",     'c', "1.0",    "FLOAT",
     "set FLOAT for cost parameter(default 1.0)" },
    {"eta",      'e', "0.0001", "FLOAT",
     "set FLOAT for termination criterion(default 0.0001)" },
    {"convert",  'C',  0,       0,
     "convert text model to binary model" },
	{"mmap",  'M',  0,       0,
	"Use precompiled binary train file" },
    {"textmodel", 't', 0,       0,
     "build also text model file for debugging" },
    {"algorithm",  'a', "CRF",   "(CRF|MIRA)", "select training algorithm" },
    {"thread", 'p',   "1",       "INT",   "number of threads(default 1)" },
    {"shrinking-size", 'H', "20", "INT",
     "set INT for number of iterations variable needs to "
     " be optimal before considered for shrinking. (default 20)" },
    {"version",  'v', 0,        0,       "show the version and exit" },
    {"help",     'h', 0,        0,       "show this help and exit" },
    {0, 0, 0, 0, 0}
  };

  CRFPP::Param param;

  param.open(argc, argv, long_options);

  if (!param.help_version()) return 0;

  bool convert = param.get<bool>("convert");
  bool bmmap = param.get<bool>("mmap");

  const std::vector<std::string> &rest = param.rest_args();
  if (param.get<bool>("help") ||
      (convert && rest.size() != 2) || (!convert && rest.size() != 3)) {
    std::cout << param.help();
    return 0;
  }

  size_t         freq           = param.get<int>("freq");
  size_t         maxiter        = param.get<int>("maxiter");
  double         C              = param.get<float>("cost");
  double         eta            = param.get<float>("eta");
  bool           textmodel      = param.get<bool>("textmodel");
  unsigned short thread         = param.get<unsigned short>("thread");
  unsigned short shrinking_size = param.get<unsigned short>("shrinking-size");
  std::string salgo = param.get<std::string>("algorithm");

  toLower(&salgo);

  int algorithm = CRFPP::Encoder::MIRA;
  if (salgo == "crf" || salgo == "crf-l2") {
    algorithm = CRFPP::Encoder::CRF_L2;
  } else if (salgo == "crf-l1") {
    algorithm = CRFPP::Encoder::CRF_L1;
  } else if (salgo == "mira") {
    algorithm = CRFPP::Encoder::MIRA;
  } else if (salgo == "mmap") {
	  algorithm = CRFPP::Encoder::MMAP;
  }else{
    std::cerr << "unknown alogrithm: " << salgo << std::endl;
    return -1;
  }

  CRFPP::Encoder encoder;
  if (convert) {
    if (!encoder.convert(rest[0].c_str(), rest[1].c_str())) {
      std::cerr << encoder.what() << std::endl;
      return -1;
    }
  } else if(bmmap){
	if (!encoder.train(rest[0].c_str(),
		  rest[2].c_str(),
		  textmodel,
		  maxiter, freq, eta, C, thread, shrinking_size,
		  algorithm)) {
			  std::cerr << encoder.what() << std::endl;
			  return -1;
	}
  }else{
    if (!encoder.learn(rest[0].c_str(),
                       rest[1].c_str(),
                       rest[2].c_str(),
                       textmodel,
                       maxiter, freq, eta, C, thread, shrinking_size,
                       algorithm)) {
      std::cerr << encoder.what() << std::endl;
      return -1;
    }
  }

  return 0;
}
