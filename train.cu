// cpu_autoencoder.cu
// CPU-only implementation of conv autoencoder (forward + backward + training)
// Compile: nvcc -O2 cpu_autoencoder.cu -o cpu_autoencoder
// Run: ./cpu_autoencoder /path/to/cifar-10-batches-bin

#include <bits/stdc++.h>
#include <chrono>
using namespace std;
using float32 = float;

// ------------------------ Utilities & Tensor4D ------------------------
struct Tensor4D {
    int N, C, H, W;
    vector<float32> data;
    Tensor4D() : N(0), C(0), H(0), W(0) {}
    Tensor4D(int n,int c,int h,int w) : N(n),C(c),H(h),W(w),data(n*c*h*w, 0.0f) {}
    inline float32& at(int n,int c,int h,int w) {
        return data[ ((n*C + c)*H + h)*W + w ];
    }
    inline const float32& at(int n,int c,int h,int w) const {
        return data[ ((n*C + c)*H + h)*W + w ];
    }
    void fill(float32 v){ std::fill(data.begin(), data.end(), v); }
};

static inline int idx4(int N,int C,int H,int W,int n,int c,int h,int w){
    return ((n*C + c)*H + h)*W + w;
}

float32 frand(float32 a= -0.08f, float32 b=0.08f){
    return a + static_cast<float32>(rand())/RAND_MAX*(b-a);
}

// ------------------------ CIFAR-10 loader ------------------------
bool load_cifar_batch(const string &file, vector<vector<float32>>& images, vector<int>& labels){
    FILE *f = fopen(file.c_str(),"rb");
    if(!f) return false;
    const int record_size = 1 + 3072;
    unsigned char buffer[3073];
    while(fread(buffer, 1, record_size, f) == record_size){
        int label = buffer[0];
        vector<float32> img(3072);
        for(int i=0;i<3072;i++) img[i] = buffer[1+i] / 255.0f;
        images.push_back(move(img));
        labels.push_back(label);
    }
    fclose(f);
    return true;
}

struct CIFAR10 {
    vector<vector<float32>> train_images;
    vector<int> train_labels;
    vector<vector<float32>> test_images;
    vector<int> test_labels;

    void load(const string &folder){
        train_images.clear(); train_labels.clear(); test_images.clear(); test_labels.clear();
        for(int b=1;b<=5;b++){
            string fn = folder + "/data_batch_" + to_string(b) + ".bin";
            if(!load_cifar_batch(fn, train_images, train_labels)){
                cerr << "Failed to load " << fn << endl;
                exit(1);
            }
        }
        string tf = folder + "/test_batch.bin";
        if(!load_cifar_batch(tf, test_images, test_labels)){
            cerr << "Failed to load " << tf << endl;
            exit(1);
        }
        cerr << "Loaded train: " << train_images.size() << ", test: " << test_images.size() << endl;
    }

    Tensor4D get_batch_tensor_from_train_indices(const vector<int>& indices){
        int N = indices.size(), C = 3, H = 32, W = 32;
        Tensor4D t(N,C,H,W);
        for(int i=0;i<N;i++){
            int idx = indices[i];
            const auto &img = train_images[idx];
            for(int c=0;c<3;c++){
                for(int y=0;y<32;y++){
                    for(int x=0;x<32;x++){
                        int pix = c*1024 + y*32 + x;
                        t.at(i,c,y,x) = img[pix];
                    }
                }
            }
        }
        return t;
    }

    vector<int> shuffle_indices(){
        vector<int> idx(train_images.size());
        iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), std::default_random_engine(rand()));
        return idx;
    }
};

// ------------------------ Layers ------------------------

// Conv2D (3x3, padding, stride=1)
struct Conv2D {
    int in_channels, out_channels, kernel_size, pad;
    vector<float32> weight; // out_ch * in_ch * k * k
    vector<float32> bias;
    vector<float32> grad_w;
    vector<float32> grad_b;
    Tensor4D input_cache;

    Conv2D(int in_ch=1, int out_ch=1, int k=3, int padding=1) :
        in_channels(in_ch), out_channels(out_ch), kernel_size(k), pad(padding)
    {
        weight.resize(out_ch * in_ch * k * k);
        bias.resize(out_ch);
        grad_w.resize(weight.size());
        grad_b.resize(bias.size());
        for(auto &v: weight) v = frand(-0.08f, 0.08f);
        for(auto &b: bias) b = 0.0f;
    }

    Tensor4D forward(const Tensor4D &x){
        input_cache = x;
        int N = x.N, H = x.H, W = x.W;
        Tensor4D out(N, out_channels, H, W);
        out.fill(0.0f);
        for(int n=0;n<N;n++){
            for(int oc=0; oc<out_channels; oc++){
                for(int oh=0; oh<H; oh++){
                    for(int ow=0; ow<W; ow++){
                        float32 val = 0.0f;
                        for(int ic=0; ic<in_channels; ic++){
                            for(int kh=0; kh<kernel_size; kh++){
                                for(int kw=0; kw<kernel_size; kw++){
                                    int ih = oh + kh - pad;
                                    int iw = ow + kw - pad;
                                    float32 in_v = 0.0f;
                                    if(ih>=0 && ih<H && iw>=0 && iw<W) in_v = x.at(n,ic,ih,iw);
                                    int widx = ((oc*in_channels + ic)*kernel_size + kh)*kernel_size + kw;
                                    val += in_v * weight[widx];
                                }
                            }
                        }
                        val += bias[oc];
                        out.at(n,oc,oh,ow) = val;
                    }
                }
            }
        }
        return out;
    }

    Tensor4D backward(const Tensor4D &grad_output, float32 lr){
        int N = grad_output.N, H = grad_output.H, W = grad_output.W;
        std::fill(grad_w.begin(), grad_w.end(), 0.0f);
        std::fill(grad_b.begin(), grad_b.end(), 0.0f);
        Tensor4D grad_input(N, in_channels, H, W);
        grad_input.fill(0.0f);

        for(int n=0;n<N;n++){
            for(int oc=0; oc<out_channels; oc++){
                for(int oh=0; oh<H; oh++){
                    for(int ow=0; ow<W; ow++){
                        float32 go = grad_output.at(n,oc,oh,ow);
                        grad_b[oc] += go;
                        for(int ic=0; ic<in_channels; ic++){
                            for(int kh=0; kh<kernel_size; kh++){
                                for(int kw=0; kw<kernel_size; kw++){
                                    int ih = oh + kh - pad;
                                    int iw = ow + kw - pad;
                                    float32 in_v = 0.0f;
                                    if(ih>=0 && ih<input_cache.H && iw>=0 && iw<input_cache.W) in_v = input_cache.at(n,ic,ih,iw);
                                    int widx = ((oc*in_channels + ic)*kernel_size + kh)*kernel_size + kw;
                                    grad_w[widx] += in_v * go;
                                    if(ih>=0 && ih<input_cache.H && iw>=0 && iw<input_cache.W){
                                        grad_input.at(n,ic,ih,iw) += weight[widx] * go;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        float32 invN = 1.0f / N;
        for(auto &v: grad_w) v *= invN;
        for(auto &v: grad_b) v *= invN;
        for(size_t i=0;i<weight.size();i++) weight[i] -= lr * grad_w[i];
        for(size_t i=0;i<bias.size();i++) bias[i] -= lr * grad_b[i];
        return grad_input;
    }

    void save(const string &prefix){
        string fn = prefix + "_w.bin";
        FILE *f = fopen(fn.c_str(),"wb");
        if(f){
            fwrite(weight.data(), sizeof(float32), weight.size(), f);
            fclose(f);
        }
        string fn2 = prefix + "_b.bin";
        f = fopen(fn2.c_str(),"wb");
        if(f){
            fwrite(bias.data(), sizeof(float32), bias.size(), f);
            fclose(f);
        }
    }
    void load(const string &prefix){
        string fn = prefix + "_w.bin";
        FILE *f = fopen(fn.c_str(),"rb");
        if(f){ fread(weight.data(), sizeof(float32), weight.size(), f); fclose(f); }
        string fn2 = prefix + "_b.bin";
        f = fopen(fn2.c_str(),"rb");
        if(f){ fread(bias.data(), sizeof(float32), bias.size(), f); fclose(f); }
    }
};

// ReLU
struct ReLU {
    Tensor4D input_cache;
    Tensor4D forward(const Tensor4D &x){
        input_cache = x;
        Tensor4D out = x;
        for(size_t i=0;i<out.data.size();i++){
            if(out.data[i] <= 0.0f) out.data[i] = 0.0f;
        }
        return out;
    }
    Tensor4D backward(const Tensor4D &grad_output){
        Tensor4D g = grad_output;
        for(size_t i=0;i<g.data.size();i++){
            if(input_cache.data[i] <= 0.0f) g.data[i] = 0.0f;
        }
        return g;
    }
};

// MaxPool 2x2 stride 2 (with argmax)
struct MaxPool2D {
    int kernel, stride;
    vector<int> argmax_idx;
    int inN, inC, inH, inW, outH, outW;
    MaxPool2D(int k=2, int s=2): kernel(k), stride(s) {}
    Tensor4D forward(const Tensor4D &x){
        inN = x.N; inC = x.C; inH = x.H; inW = x.W;
        outH = inH / kernel; outW = inW / kernel;
        Tensor4D out(inN, inC, outH, outW);
        argmax_idx.assign(inN*inC*outH*outW, -1);
        for(int n=0;n<inN;n++){
            for(int c=0;c<inC;c++){
                for(int oh=0; oh<outH; oh++){
                    for(int ow=0; ow<outW; ow++){
                        float32 best = -1e9;
                        int bestidx = -1;
                        for(int kh=0; kh<kernel; kh++){
                            for(int kw=0; kw<kernel; kw++){
                                int ih = oh*stride + kh;
                                int iw = ow*stride + kw;
                                float32 v = x.at(n,c,ih,iw);
                                if(v > best){ best = v; bestidx = idx4(inN,inC,inH,inW,n,c,ih,iw); }
                            }
                        }
                        out.at(n,c,oh,ow) = best;
                        int flat = ((n*inC + c)*outH + oh)*outW + ow;
                        argmax_idx[flat] = bestidx;
                    }
                }
            }
        }
        return out;
    }
    Tensor4D backward(const Tensor4D &grad_output){
        Tensor4D grad(inN, inC, inH, inW);
        grad.fill(0.0f);
        for(int n=0;n<inN;n++){
            for(int c=0;c<inC;c++){
                for(int oh=0; oh<outH; oh++){
                    for(int ow=0; ow<outW; ow++){
                        int flat = ((n*inC + c)*outH + oh)*outW + ow;
                        int src = argmax_idx[flat];
                        if(src>=0){
                            int tmp = src;
                            int w = tmp % inW; tmp /= inW;
                            int h = tmp % inH; tmp /= inH;
                            int ch = tmp % inC; tmp /= inC;
                            int nn = tmp;
                            grad.at(nn, ch, h, w) += grad_output.at(n,c,oh,ow);
                        }
                    }
                }
            }
        }
        return grad;
    }
};

// Upsample nearest x2
struct Upsample2D {
    Tensor4D input_cache;
    int scale;
    Upsample2D(int s=2): scale(s){}
    Tensor4D forward(const Tensor4D &x){
        input_cache = x;
        int N=x.N,C=x.C,H=x.H,W=x.W;
        Tensor4D out(N,C,H*scale,W*scale);
        for(int n=0;n<N;n++){
            for(int c=0;c<C;c++){
                for(int h=0;h<H;h++){
                    for(int w=0;w<W;w++){
                        float32 v = x.at(n,c,h,w);
                        for(int sh=0;sh<scale;sh++){
                            for(int sw=0;sw<scale;sw++){
                                out.at(n,c,h*scale+sh,w*scale+sw) = v;
                            }
                        }
                    }
                }
            }
        }
        return out;
    }
    Tensor4D backward(const Tensor4D &grad_output){
        int N=input_cache.N,C=input_cache.C,H=input_cache.H,W=input_cache.W;
        Tensor4D grad(N,C,H,W);
        grad.fill(0.0f);
        for(int n=0;n<N;n++){
            for(int c=0;c<C;c++){
                for(int h=0;h<H;h++){
                    for(int w=0;w<W;w++){
                        float32 s = 0.0f;
                        for(int sh=0; sh<scale; sh++){
                            for(int sw=0; sw<scale; sw++){
                                s += grad_output.at(n,c,h*scale+sh,w*scale+sw);
                            }
                        }
                        grad.at(n,c,h,w) = s;
                    }
                }
            }
        }
        return grad;
    }
};

// MSE Loss
struct MSELoss {
    Tensor4D target_cache;
    float32 forward(const Tensor4D &pred, const Tensor4D &target){
        target_cache = target;
        double s=0.0;
        for(size_t i=0;i<pred.data.size();i++){
            double d = pred.data[i] - target.data[i];
            s += d*d;
        }
        return static_cast<float32>(s / pred.data.size());
    }
    Tensor4D backward(const Tensor4D &pred){
        Tensor4D g = pred;
        int n = pred.data.size();
        for(int i=0;i<n;i++){
            g.data[i] = 2.0f * (pred.data[i] - target_cache.data[i]) / n;
        }
        return g;
    }
};

// ------------------------ Autoencoder ------------------------
struct Autoencoder {
    Conv2D conv1; // 3 -> 32
    Conv2D conv2; // 32 -> 64
    Conv2D conv3; // 64 -> 128 (latent)

    Conv2D deconv1; // 128 -> 64
    Conv2D deconv2; // 64 -> 32
    Conv2D deconv3; // 32 -> 3

    ReLU relu;
    MaxPool2D pool;
    Upsample2D upsample;
    MSELoss loss;

    // caches
    Tensor4D a1, a1p; // after conv1, after pool1
    Tensor4D a2, a2p; // after conv2, after pool2
    Tensor4D z; // after conv3 (latent)
    Tensor4D recon; // reconstructed output

    Autoencoder() :
        conv1(3,32,3,1),
        conv2(32,64,3,1),
        conv3(64,128,3,1),
        deconv1(128,64,3,1),
        deconv2(64,32,3,1),
        deconv3(32,3,3,1),
        pool(2,2),
        upsample(2)
    {}

    Tensor4D forward(const Tensor4D &x){
        // encoder
        a1 = conv1.forward(x);
        Tensor4D r1 = relu.forward(a1);
        a1p = pool.forward(r1);

        a2 = conv2.forward(a1p);
        Tensor4D r2 = relu.forward(a2);
        a2p = pool.forward(r2);

        z = conv3.forward(a2p);
        Tensor4D rz = relu.forward(z); // latent activation

        // decoder
        Tensor4D up1 = upsample.forward(rz); // 8->16
        Tensor4D d1 = deconv1.forward(up1);
        Tensor4D rd1 = relu.forward(d1);

        Tensor4D up2 = upsample.forward(rd1); // 16->32
        Tensor4D d2 = deconv2.forward(up2);
        Tensor4D rd2 = relu.forward(d2);

        recon = deconv3.forward(rd2);
        // final activation: clamp to [0,1] (since input normalized)
        Tensor4D out = recon;
        for(size_t i=0;i<out.data.size();i++){
            if(out.data[i] < 0.0f) out.data[i] = 0.0f;
            if(out.data[i] > 1.0f) out.data[i] = 1.0f;
        }
        return out;
    }

    float32 compute_loss_and_backward(const Tensor4D &input, float32 lr){
        Tensor4D pred = forward(input);
        float32 L = loss.forward(pred, input);
        // backward pass
        Tensor4D grad = loss.backward(pred); // dL/dpred

        // decoder backward
        Tensor4D g_deconv3 = deconv3.backward(grad, lr); // returns grad wrt rd2
        Tensor4D g_rd2 = relu.backward(g_deconv3);
        Tensor4D g_deconv2 = deconv2.backward(g_rd2, lr);
        Tensor4D g_up2 = upsample.backward(g_deconv2);
        Tensor4D g_rd1 = relu.backward(g_up2);
        Tensor4D g_deconv1 = deconv1.backward(g_rd1, lr);
        Tensor4D g_up1 = upsample.backward(g_deconv1);
        Tensor4D g_rz = relu.backward(g_up1);
        // encoder backward
        Tensor4D g_conv3 = conv3.backward(g_rz, lr);
        Tensor4D g_pool2 = pool.backward(g_conv3);
        Tensor4D g_r2_back = relu.backward(g_pool2);
        Tensor4D g_conv2 = conv2.backward(g_r2_back, lr);
        Tensor4D g_pool1 = pool.backward(g_conv2);
        Tensor4D g_r1_back = relu.backward(g_pool1);
        Tensor4D g_conv1 = conv1.backward(g_r1_back, lr);
        // convX.backward already updated weights using lr
        return L;
    }

    void save_weights_for_epoch(int epoch){
        string p = "model_epoch_" + to_string(epoch);
        conv1.save(p + "_conv1");
        conv2.save(p + "_conv2");
        conv3.save(p + "_conv3");
        deconv1.save(p + "_deconv1");
        deconv2.save(p + "_deconv2");
        deconv3.save(p + "_deconv3");
    }

    void load_weights_prefix(const string &prefix){
        conv1.load(prefix + "_conv1");
        conv2.load(prefix + "_conv2");
        conv3.load(prefix + "_conv3");
        deconv1.load(prefix + "_deconv1");
        deconv2.load(prefix + "_deconv2");
        deconv3.load(prefix + "_deconv3");
    }
};

// ------------------------ Training Loop ------------------------
int main(int argc, char** argv){
    srand(1234);
    if(argc < 2){
        cerr << "Usage: " << argv[0] << " /path/to/cifar-10-batches-bin" << endl;
        return 1;
    }
    string data_dir = argv[1];
    CIFAR10 dataset;
    dataset.load(data_dir);

    Autoencoder model;

    const int batch_size = 32;
    const int epochs = 20;
    const float32 lr = 0.001f;

    int num_samples = dataset.train_images.size();
    int steps_per_epoch = (num_samples + batch_size - 1) / batch_size;

    cerr << "Training: epochs=" << epochs << " batch_size=" << batch_size << " steps/epoch~" << steps_per_epoch << endl;

    for(int e=1;e<=epochs;e++){
        auto t0 = chrono::high_resolution_clock::now();
        vector<int> order = dataset.shuffle_indices();
        double epoch_loss = 0.0;
        int seen = 0;
        for(int s=0; s<steps_per_epoch; s++){
            int start = s * batch_size;
            int end = min(start + batch_size, num_samples);
            if(start >= end) break;
            vector<int> batch_idx;
            batch_idx.reserve(end-start);
            for(int i=start;i<end;i++) batch_idx.push_back(order[i]);
            Tensor4D batch = dataset.get_batch_tensor_from_train_indices(batch_idx);
            float32 L = model.compute_loss_and_backward(batch, lr);
            epoch_loss += L * (end - start);
            seen += (end - start);

            if((s+1) % 200 == 0 || s==steps_per_epoch-1){
                cerr << "[Epoch " << e << "] step " << (s+1) << "/" << steps_per_epoch << " partial loss=" << L << endl;
            }
        }
        epoch_loss /= seen;
        auto t1 = chrono::high_resolution_clock::now();
        double sec = chrono::duration<double>(t1 - t0).count();
        cerr << "Epoch " << e << " finished. avg loss=" << epoch_loss << " time=" << sec << "s" << endl;
        // save weights after each epoch
        model.save_weights_for_epoch(e);
    }

    cerr << "Training completed. Final model saved as model_epoch_<epoch>_*.bin" << endl;
    return 0;
}
