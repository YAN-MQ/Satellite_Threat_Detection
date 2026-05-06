/*
 * Level 4B runtime: ns-3 internal federated training with libtorch.
 *
 * This program keeps the federated round scheduler, topology evolution,
 * training, aggregation and evaluation inside one C++/ns-3 executable.
 */

#include "ns3/core-module.h"

#include <torch/torch.h>
#include <torch/cuda.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace ns3;

namespace
{

struct RuntimeConfig
{
    std::string dataDir;
    std::string outputDir = "4_train/experiments/OrbitShield_FL_ns3_libtorch/cicids17";
    std::string initStateDir;
    std::string dataset = "cicids17";
    uint32_t numClients = 12;
    uint32_t numPlanes = 3;
    uint32_t rounds = 20;
    uint32_t localEpochs = 1;
    uint32_t batchSize = 512;
    uint32_t inputDim = 18;
    uint32_t seqLen = 10;
    uint32_t numClasses = 3;
    uint32_t trainSamples = 0;
    uint32_t valSamples = 0;
    uint32_t testSamples = 0;
    uint32_t convDim = 16;
    uint32_t dscDim = 48;
    uint32_t hiddenDim = 64;
    bool bidirectional = false;
    uint32_t fcHidden = 64;
    double dropout = 0.3;
    double lr = 1e-3;
    double weightDecay = 1e-2;
    double beta = 0.1;
    double betaFloor = 0.05;
    double lambdaS = 0.1;
    double rho = 0.5;
    double mu = 0.8;
    double globalMomentum = 0.1;
    double rMin = 0.1;
    double scoreSimWeight = 0.4;
    double scoreImproveWeight = 0.4;
    double scoreStableWeight = 0.2;
    double intraSuccessProb = 0.98;
    double interSuccessProb = 0.75;
    double interLoss = 0.05;
    double interDelayMs = 25.0;
    double intraDelayMs = 10.0;
    double intraBandwidthMbps = 500.0;
    double interBandwidthMbps = 120.0;
    double roundDurationSeconds = 30.0;
    uint32_t contactPeriod = 4;
    uint32_t contactDurationRounds = 2;
    uint32_t warmupRounds = 2;
    uint32_t seed = 42;
    std::string device = "cuda";
};

struct LinkState
{
    bool available = false;
    bool success = false;
    double delayMs = 0.0;
    double bandwidthMbps = 0.0;
    double packetLoss = 1.0;
    double contactDurationSeconds = 0.0;
};

struct EvalMetrics
{
    double loss = 0.0;
    double accuracy = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
    std::vector<std::vector<int64_t>> confusionMatrix;
};

struct ClientState
{
    uint32_t clientId = 0;
    uint32_t planeId = 0;
    torch::Tensor indices;
    double reputation = 1.0;
    uint32_t lastSyncRound = 0;
    uint32_t participationCount = 0;
    uint32_t successfulSyncCount = 0;
};

struct DscCbamGruImpl;

struct ClientUploadPayload
{
    uint32_t clientId = 0;
    uint32_t planeId = 0;
    std::shared_ptr<DscCbamGruImpl> weights;
    std::shared_ptr<DscCbamGruImpl> baseModel;
    double averageLoss = 0.0;
    int64_t sampleCount = 0;
    double reputation = 1.0;
    uint32_t lastSyncRound = 0;
    double linkQuality = 0.0;
};

struct StateTensorRecord
{
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    std::string fileName;
};

std::string
Quote(const std::string &value)
{
    std::ostringstream oss;
    oss << '"';
    for (char c : value)
    {
        switch (c)
        {
        case '\\':
            oss << "\\\\";
            break;
        case '"':
            oss << "\\\"";
            break;
        case '\n':
            oss << "\\n";
            break;
        default:
            oss << c;
            break;
        }
    }
    oss << '"';
    return oss.str();
}

double
EstimateLinkQuality(const LinkState &state)
{
    if (!state.available || !state.success)
    {
        return 0.0;
    }
    const double delayPenalty = 1.0 / (1.0 + (state.delayMs / 1000.0));
    const double lossPenalty = std::max(0.0, 1.0 - state.packetLoss);
    return std::clamp(delayPenalty * lossPenalty, 0.0, 1.0);
}

double
BoundedImprovement(double lossBefore, double lossAfter)
{
    const double delta = lossBefore - lossAfter;
    return (std::tanh(delta) + 1.0) / 2.0;
}

double
TensorBytes(const torch::nn::Module &module)
{
    int64_t total = 0;
    for (const auto &item : module.named_parameters(true))
    {
        total += item.value().numel() * item.value().element_size();
    }
    for (const auto &item : module.named_buffers(true))
    {
        total += item.value().numel() * item.value().element_size();
    }
    return static_cast<double>(total);
}

double
EffectiveBandwidthMbps(const LinkState &state)
{
    return std::max(0.0, state.bandwidthMbps * (1.0 - state.packetLoss));
}

bool
CanTransfer(const torch::nn::Module &module, const LinkState &state)
{
    if (!state.available || !state.success)
    {
        return false;
    }
    const double bw = EffectiveBandwidthMbps(state);
    if (bw <= 0.0)
    {
        return false;
    }
    const double transferSeconds = (TensorBytes(module) * 8.0) / (bw * 1e6);
    return transferSeconds <= state.contactDurationSeconds;
}

struct DepthwiseSeparableConv1dImpl : torch::nn::Module
{
    torch::nn::Conv1d dw{nullptr};
    torch::nn::Conv1d pw{nullptr};
    torch::nn::BatchNorm1d bn{nullptr};

    DepthwiseSeparableConv1dImpl(int64_t cIn, int64_t cOut)
    {
        dw = register_module("dw", torch::nn::Conv1d(torch::nn::Conv1dOptions(cIn, cIn, 3).padding(1).groups(cIn)));
        pw = register_module("pw", torch::nn::Conv1d(torch::nn::Conv1dOptions(cIn, cOut, 1)));
        bn = register_module("bn", torch::nn::BatchNorm1d(cOut));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = dw->forward(x);
        x = pw->forward(x);
        x = bn->forward(x);
        return torch::relu(x);
    }
};
TORCH_MODULE(DepthwiseSeparableConv1d);

struct ChannelAttentionImpl : torch::nn::Module
{
    torch::nn::AdaptiveAvgPool1d avgPool{nullptr};
    torch::nn::AdaptiveMaxPool1d maxPool{nullptr};
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    int64_t channels = 0;

    explicit ChannelAttentionImpl(int64_t c, int64_t r = 8) : channels(c)
    {
        const int64_t reduced = std::max<int64_t>(1, c / r);
        avgPool = register_module("avg_pool", torch::nn::AdaptiveAvgPool1d(1));
        maxPool = register_module("max_pool", torch::nn::AdaptiveMaxPool1d(1));
        fc1 = register_module("fc1", torch::nn::Linear(torch::nn::LinearOptions(c, reduced).bias(false)));
        fc2 = register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(reduced, c).bias(false)));
    }

    torch::Tensor Project(torch::Tensor pooled)
    {
        auto y = pooled.view({pooled.size(0), channels});
        y = torch::relu(fc1->forward(y));
        y = fc2->forward(y);
        return y;
    }

    torch::Tensor forward(torch::Tensor x)
    {
        auto avgOut = Project(avgPool->forward(x));
        auto maxOut = Project(maxPool->forward(x));
        auto attn = torch::sigmoid(avgOut + maxOut).view({x.size(0), channels, 1});
        return x * attn;
    }
};
TORCH_MODULE(ChannelAttention);

struct SpatialAttentionImpl : torch::nn::Module
{
    torch::nn::Conv1d conv{nullptr};

    SpatialAttentionImpl()
    {
        conv = register_module("conv", torch::nn::Conv1d(torch::nn::Conv1dOptions(2, 1, 7).padding(3).bias(false)));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        auto avgOut = torch::mean(x, 1, true);
        auto maxOut = std::get<0>(torch::max(x, 1, true));
        auto attn = torch::sigmoid(conv->forward(torch::cat({avgOut, maxOut}, 1)));
        return x * attn;
    }
};
TORCH_MODULE(SpatialAttention);

struct CBAMImpl : torch::nn::Module
{
    ChannelAttention channel{nullptr};
    SpatialAttention spatial{nullptr};

    explicit CBAMImpl(int64_t c)
    {
        channel = register_module("channel", ChannelAttention(c));
        spatial = register_module("spatial", SpatialAttention());
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = channel->forward(x);
        x = spatial->forward(x);
        return x;
    }
};
TORCH_MODULE(CBAM);

struct DscCbamGruImpl : torch::nn::Module
{
    torch::nn::Conv1d conv{nullptr};
    DepthwiseSeparableConv1d dsc{nullptr};
    CBAM cbam{nullptr};
    torch::nn::GRU gru{nullptr};
    torch::nn::Linear fc1{nullptr};
    torch::nn::Dropout drop{nullptr};
    torch::nn::Linear fc2{nullptr};
    bool bidirectional = false;

    DscCbamGruImpl(int64_t inputDim,
                   int64_t numClasses,
                   int64_t convDim,
                   int64_t dscDim,
                   int64_t hiddenDim,
                   bool bi,
                   int64_t fcHidden,
                   double dropout)
        : bidirectional(bi)
    {
        const int64_t gruOut = hiddenDim * (bi ? 2 : 1);
        conv = register_module("conv", torch::nn::Conv1d(torch::nn::Conv1dOptions(inputDim, convDim, 1)));
        dsc = register_module("dsc", DepthwiseSeparableConv1d(convDim, dscDim));
        cbam = register_module("cbam", CBAM(dscDim));
        gru = register_module(
            "gru",
            torch::nn::GRU(torch::nn::GRUOptions(dscDim, hiddenDim).batch_first(true).bidirectional(bi)));
        fc1 = register_module("fc1", torch::nn::Linear(gruOut, fcHidden));
        drop = register_module("drop", torch::nn::Dropout(dropout));
        fc2 = register_module("fc2", torch::nn::Linear(fcHidden, numClasses));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = x.permute({0, 2, 1});
        x = conv->forward(x);
        x = dsc->forward(x);
        x = cbam->forward(x);
        x = x.permute({0, 2, 1});
        auto gruOut = std::get<0>(gru->forward(x));
        x = gruOut.index({torch::indexing::Slice(), -1, torch::indexing::Slice()});
        x = torch::relu(fc1->forward(x));
        x = drop->forward(x);
        return fc2->forward(x);
    }
};
TORCH_MODULE(DscCbamGru);

std::shared_ptr<DscCbamGruImpl>
CreateModel(const RuntimeConfig &config, const torch::Device &device)
{
    auto model = std::make_shared<DscCbamGruImpl>(
        config.inputDim,
        config.numClasses,
        config.convDim,
        config.dscDim,
        config.hiddenDim,
        config.bidirectional,
        config.fcHidden,
        config.dropout);
    model->to(device);
    return model;
}

void
CopyState(const torch::nn::Module &src, torch::nn::Module &dst)
{
    torch::NoGradGuard noGrad;
    auto srcParams = src.named_parameters(true);
    auto dstParams = dst.named_parameters(true);
    for (const auto &item : srcParams)
    {
        dstParams[item.key()].detach().copy_(item.value().detach());
    }
    auto srcBuffers = src.named_buffers(true);
    auto dstBuffers = dst.named_buffers(true);
    for (const auto &item : srcBuffers)
    {
        dstBuffers[item.key()].detach().copy_(item.value().detach());
    }
}

std::shared_ptr<DscCbamGruImpl>
CloneModel(const DscCbamGruImpl &source, const RuntimeConfig &config, const torch::Device &device)
{
    auto clone = CreateModel(config, device);
    CopyState(source, *clone);
    return clone;
}

torch::Tensor
FlattenDelta(const torch::nn::Module &after, const torch::nn::Module &before)
{
    std::vector<torch::Tensor> chunks;
    auto afterParams = after.named_parameters(true);
    auto beforeParams = before.named_parameters(true);
    for (const auto &item : afterParams)
    {
        chunks.push_back((item.value().detach().cpu() - beforeParams[item.key()].detach().cpu()).reshape(-1).to(torch::kFloat32));
    }
    auto afterBuffers = after.named_buffers(true);
    auto beforeBuffers = before.named_buffers(true);
    for (const auto &item : afterBuffers)
    {
        if (!item.value().is_floating_point())
        {
            continue;
        }
        chunks.push_back((item.value().detach().cpu() - beforeBuffers[item.key()].detach().cpu()).reshape(-1).to(torch::kFloat32));
    }
    if (chunks.empty())
    {
        return torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32));
    }
    return torch::cat(chunks);
}

double
CosineSimilarityDelta(
    const torch::nn::Module &afterA,
    const torch::nn::Module &beforeA,
    const torch::nn::Module &afterB,
    const torch::nn::Module &beforeB)
{
    const auto vecA = FlattenDelta(afterA, beforeA);
    const auto vecB = FlattenDelta(afterB, beforeB);
    const double normA = vecA.norm().item<double>();
    const double normB = vecB.norm().item<double>();
    if (normA <= 0.0 || normB <= 0.0)
    {
        return 0.0;
    }
    return torch::dot(vecA, vecB).item<double>() / (normA * normB);
}

std::vector<std::string>
Split(const std::string &value, char delimiter)
{
    std::vector<std::string> parts;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, delimiter))
    {
        parts.push_back(item);
    }
    return parts;
}

std::vector<int64_t>
ParseShape(const std::string &shapeSpec)
{
    std::vector<int64_t> shape;
    if (shapeSpec.empty())
    {
        return shape;
    }
    for (const auto &part : Split(shapeSpec, ','))
    {
        if (!part.empty())
        {
            shape.push_back(std::stoll(part));
        }
    }
    return shape;
}

torch::Tensor
LoadTypedTensorRaw(const std::filesystem::path &path, const std::string &dtype, const std::vector<int64_t> &shape)
{
    int64_t numel = 1;
    for (int64_t dim : shape)
    {
        numel *= dim;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        throw std::runtime_error("Failed to open tensor file: " + path.string());
    }

    if (dtype == "float32")
    {
        std::vector<float> buffer(static_cast<size_t>(numel));
        in.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(numel * sizeof(float)));
        if (!in)
        {
            throw std::runtime_error("Failed to read float32 tensor: " + path.string());
        }
        return torch::from_blob(buffer.data(), shape, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    }
    if (dtype == "float64")
    {
        std::vector<double> buffer(static_cast<size_t>(numel));
        in.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(numel * sizeof(double)));
        if (!in)
        {
            throw std::runtime_error("Failed to read float64 tensor: " + path.string());
        }
        return torch::from_blob(buffer.data(), shape, torch::TensorOptions().dtype(torch::kFloat64)).clone();
    }
    if (dtype == "int64")
    {
        std::vector<int64_t> buffer(static_cast<size_t>(numel));
        in.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(numel * sizeof(int64_t)));
        if (!in)
        {
            throw std::runtime_error("Failed to read int64 tensor: " + path.string());
        }
        return torch::from_blob(buffer.data(), shape, torch::TensorOptions().dtype(torch::kInt64)).clone();
    }
    if (dtype == "int32")
    {
        std::vector<int32_t> buffer(static_cast<size_t>(numel));
        in.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(numel * sizeof(int32_t)));
        if (!in)
        {
            throw std::runtime_error("Failed to read int32 tensor: " + path.string());
        }
        return torch::from_blob(buffer.data(), shape, torch::TensorOptions().dtype(torch::kInt32)).clone();
    }
    throw std::runtime_error("Unsupported dtype in state manifest: " + dtype);
}

std::vector<StateTensorRecord>
LoadStateManifest(const std::filesystem::path &manifestPath)
{
    std::ifstream in(manifestPath);
    if (!in)
    {
        throw std::runtime_error("Failed to open state manifest: " + manifestPath.string());
    }
    std::vector<StateTensorRecord> records;
    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty())
        {
            continue;
        }
        auto fields = Split(line, '\t');
        if (fields.size() != 4)
        {
            throw std::runtime_error("Invalid state manifest line: " + line);
        }
        records.push_back(StateTensorRecord{
            fields[0],
            fields[1],
            ParseShape(fields[2]),
            fields[3],
        });
    }
    return records;
}

void
LoadStateDirIntoModel(const std::filesystem::path &stateDir, torch::nn::Module &model)
{
    const auto manifest = LoadStateManifest(stateDir / "state_manifest.tsv");
    auto params = model.named_parameters(true);
    auto buffers = model.named_buffers(true);
    torch::NoGradGuard noGrad;
    for (const auto &record : manifest)
    {
        auto tensor = LoadTypedTensorRaw(stateDir / record.fileName, record.dtype, record.shape);
        if (params.contains(record.name))
        {
            params[record.name].detach().copy_(tensor.to(params[record.name].device(), params[record.name].scalar_type()));
            continue;
        }
        if (buffers.contains(record.name))
        {
            buffers[record.name].detach().copy_(tensor.to(buffers[record.name].device(), buffers[record.name].scalar_type()));
            continue;
        }
        throw std::runtime_error("Unknown parameter/buffer in init state: " + record.name);
    }
}

void
WeightedAverageInto(
    const std::vector<std::pair<std::shared_ptr<DscCbamGruImpl>, double>> &models,
    DscCbamGruImpl &target)
{
    NS_ABORT_MSG_IF(models.empty(), "Cannot average zero models");
    torch::NoGradGuard noGrad;
    double totalWeight = 0.0;
    for (const auto &[_, weight] : models)
    {
        totalWeight += weight;
    }
    if (totalWeight <= 0.0)
    {
        totalWeight = static_cast<double>(models.size());
    }

    auto targetParams = target.named_parameters(true);
    auto targetBuffers = target.named_buffers(true);

    for (auto &item : targetParams)
    {
        auto accum = torch::zeros_like(item.value());
        for (const auto &[model, weight] : models)
        {
            accum += model->named_parameters(true)[item.key()] * (weight / totalWeight);
        }
        item.value().detach().copy_(accum.detach());
    }
    for (auto &item : targetBuffers)
    {
        if (!item.value().is_floating_point())
        {
            item.value().detach().copy_(models.front().first->named_buffers(true)[item.key()].detach());
            continue;
        }
        auto accum = torch::zeros_like(item.value());
        for (const auto &[model, weight] : models)
        {
            accum += model->named_buffers(true)[item.key()] * (weight / totalWeight);
        }
        item.value().detach().copy_(accum.detach());
    }
}

EvalMetrics
EvaluateModel(
    DscCbamGruImpl &model,
    const torch::Tensor &features,
    const torch::Tensor &labels,
    uint32_t batchSize,
    uint32_t numClasses,
    const torch::Device &device)
{
    torch::NoGradGuard noGrad;
    model.eval();
    EvalMetrics metrics;
    metrics.confusionMatrix.assign(numClasses, std::vector<int64_t>(numClasses, 0));
    double totalLoss = 0.0;
    int64_t total = labels.size(0);
    int64_t correct = 0;
    std::vector<int64_t> support(numClasses, 0);
    std::vector<int64_t> predCount(numClasses, 0);
    std::vector<int64_t> truePositive(numClasses, 0);

    for (int64_t start = 0; start < total; start += batchSize)
    {
        const int64_t end = std::min<int64_t>(start + batchSize, total);
        auto x = features.narrow(0, start, end - start).to(device);
        auto y = labels.narrow(0, start, end - start).to(device);
        auto logits = model.forward(x);
        auto loss = torch::nn::functional::cross_entropy(logits, y);
        totalLoss += static_cast<double>(loss.detach().cpu().data_ptr<float>()[0]) * static_cast<double>(end - start);
        auto preds = logits.argmax(1).cpu();
        auto truth = y.cpu();
        auto predsPtr = preds.data_ptr<int64_t>();
        auto truthPtr = truth.data_ptr<int64_t>();
        for (int64_t i = 0; i < preds.size(0); ++i)
        {
            int64_t p = predsPtr[i];
            int64_t t = truthPtr[i];
            metrics.confusionMatrix[t][p] += 1;
            support[t] += 1;
            predCount[p] += 1;
            if (p == t)
            {
                truePositive[t] += 1;
                correct += 1;
            }
        }
    }

    const double totalD = static_cast<double>(std::max<int64_t>(1, total));
    metrics.loss = totalLoss / totalD;
    metrics.accuracy = static_cast<double>(correct) / totalD;

    double weightedPrecision = 0.0;
    double weightedRecall = 0.0;
    double weightedF1 = 0.0;
    for (uint32_t c = 0; c < numClasses; ++c)
    {
        const double tp = static_cast<double>(truePositive[c]);
        const double fp = static_cast<double>(predCount[c] - truePositive[c]);
        const double fn = static_cast<double>(support[c] - truePositive[c]);
        const double prec = (tp + fp) > 0.0 ? tp / (tp + fp) : 0.0;
        const double rec = (tp + fn) > 0.0 ? tp / (tp + fn) : 0.0;
        const double f1 = (prec + rec) > 0.0 ? (2.0 * prec * rec) / (prec + rec) : 0.0;
        const double w = static_cast<double>(support[c]) / totalD;
        weightedPrecision += w * prec;
        weightedRecall += w * rec;
        weightedF1 += w * f1;
    }
    metrics.precision = weightedPrecision;
    metrics.recall = weightedRecall;
    metrics.f1 = weightedF1;
    return metrics;
}

double
TrainOneClient(
    DscCbamGruImpl &model,
    const torch::Tensor &allFeatures,
    const torch::Tensor &allLabels,
    const torch::Tensor &indices,
    const RuntimeConfig &config,
    const torch::Device &device)
{
    auto subsetIndices = indices.to(
        torch::TensorOptions().device(allFeatures.device()).dtype(torch::kLong));
    auto subsetX = allFeatures.index_select(0, subsetIndices).to(device);
    auto subsetY = allLabels.index_select(0, subsetIndices).to(device);
    model.train();
    torch::optim::AdamW optimizer(
        model.parameters(),
        torch::optim::AdamWOptions(config.lr).weight_decay(config.weightDecay));

    std::mt19937 rng(config.seed);
    double totalLoss = 0.0;
    int64_t batchCount = 0;

    for (uint32_t epoch = 0; epoch < config.localEpochs; ++epoch)
    {
        auto perm = torch::randperm(
            subsetX.size(0),
            torch::TensorOptions().dtype(torch::kLong).device(subsetX.device()));
        auto x = subsetX.index_select(0, perm);
        auto y = subsetY.index_select(0, perm);
        for (int64_t start = 0; start < x.size(0); start += config.batchSize)
        {
            const int64_t end = std::min<int64_t>(start + config.batchSize, x.size(0));
            auto xb = x.narrow(0, start, end - start);
            auto yb = y.narrow(0, start, end - start);
            optimizer.zero_grad();
            auto logits = model.forward(xb);
            auto loss = torch::nn::functional::cross_entropy(logits, yb);
            loss.backward();
            optimizer.step();
            totalLoss += static_cast<double>(loss.detach().cpu().data_ptr<float>()[0]);
            batchCount += 1;
        }
    }
    return totalLoss / static_cast<double>(std::max<int64_t>(1, batchCount));
}

LinkState
BuildLinkState(
    bool available,
    double successProb,
    double delayMs,
    double bandwidthMbps,
    double packetLoss,
    double durationSeconds,
    std::mt19937 &rng)
{
    LinkState state;
    state.available = available;
    if (!available)
    {
        return state;
    }
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_real_distribution<double> jitter(0.9, 1.1);
    state.success = unit(rng) < successProb;
    state.delayMs = delayMs * jitter(rng);
    state.bandwidthMbps = bandwidthMbps * jitter(rng);
    state.packetLoss = std::clamp(packetLoss * jitter(rng), 0.0, 1.0);
    state.contactDurationSeconds = durationSeconds;
    return state;
}

bool
InterVisible(const RuntimeConfig &config, uint32_t roundIdx, uint32_t planeA, uint32_t planeB)
{
    const uint32_t offset = std::min(planeA, planeB);
    return ((roundIdx - 1 + offset) % config.contactPeriod) < config.contactDurationRounds;
}

void
WriteRoundCsv(
    const std::string &path,
    const std::vector<std::map<std::string, double>> &rows)
{
    std::ofstream out(path);
    out << "round,val_loss,val_accuracy,val_precision,val_recall,val_f1,test_accuracy,test_precision,test_recall,test_f1,communication_cost_mb,stale_update_ratio,link_failure_robustness\n";
    for (const auto &row : rows)
    {
        out << static_cast<int>(row.at("round")) << ','
            << row.at("val_loss") << ','
            << row.at("val_accuracy") << ','
            << row.at("val_precision") << ','
            << row.at("val_recall") << ','
            << row.at("val_f1") << ','
            << row.at("test_accuracy") << ','
            << row.at("test_precision") << ','
            << row.at("test_recall") << ','
            << row.at("test_f1") << ','
            << row.at("communication_cost_mb") << ','
            << row.at("stale_update_ratio") << ','
            << row.at("link_failure_robustness") << '\n';
    }
}

void
WriteSummary(
    const std::string &path,
    const RuntimeConfig &config,
    const EvalMetrics &bestVal,
    const EvalMetrics &test,
    uint32_t bestRound,
    const std::string &bestModelPath)
{
    std::ofstream out(path);
    out << "{\n";
    out << "  \"framework\": " << Quote("OrbitShield_FL_ns3_libtorch") << ",\n";
    out << "  \"dataset\": " << Quote(config.dataset) << ",\n";
    out << "  \"rounds\": " << config.rounds << ",\n";
    out << "  \"best_round\": " << bestRound << ",\n";
    out << "  \"best_val_accuracy\": " << std::fixed << std::setprecision(6) << bestVal.accuracy << ",\n";
    out << "  \"test_accuracy\": " << test.accuracy << ",\n";
    out << "  \"test_precision\": " << test.precision << ",\n";
    out << "  \"test_recall\": " << test.recall << ",\n";
    out << "  \"test_f1\": " << test.f1 << ",\n";
    out << "  \"best_model_path\": " << Quote(bestModelPath) << ",\n";
    out << "  \"confusion_matrix\": [\n";
    for (size_t i = 0; i < test.confusionMatrix.size(); ++i)
    {
        out << "    [";
        for (size_t j = 0; j < test.confusionMatrix[i].size(); ++j)
        {
            out << test.confusionMatrix[i][j];
            if (j + 1 < test.confusionMatrix[i].size())
            {
                out << ", ";
            }
        }
        out << "]";
        if (i + 1 < test.confusionMatrix.size())
        {
            out << ",";
        }
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";
}

void
WriteReputationHistory(
    const std::string &path,
    const std::vector<std::vector<double>> &history)
{
    std::ofstream out(path);
    out << "{\n";
    for (size_t clientId = 0; clientId < history.size(); ++clientId)
    {
        out << "  " << Quote("sat_" + std::to_string(clientId)) << ": [";
        for (size_t idx = 0; idx < history[clientId].size(); ++idx)
        {
            out << std::fixed << std::setprecision(6) << history[clientId][idx];
            if (idx + 1 < history[clientId].size())
            {
                out << ", ";
            }
        }
        out << "]";
        if (clientId + 1 < history.size())
        {
            out << ",";
        }
        out << "\n";
    }
    out << "}\n";
}

torch::Tensor
LoadFloatTensor(const std::string &path, const std::vector<int64_t> &shape)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        throw std::runtime_error("Failed to open " + path);
    }
    int64_t numel = 1;
    for (int64_t dim : shape)
    {
        numel *= dim;
    }
    std::vector<float> buffer(static_cast<size_t>(numel));
    in.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(numel * sizeof(float)));
    if (!in)
    {
        throw std::runtime_error("Failed to read float tensor from " + path);
    }
    return torch::from_blob(buffer.data(), shape, torch::TensorOptions().dtype(torch::kFloat32)).clone();
}

torch::Tensor
LoadLongTensorFlat(const std::string &path)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
    {
        throw std::runtime_error("Failed to open " + path);
    }
    const auto bytes = in.tellg();
    in.seekg(0, std::ios::beg);
    const int64_t count = static_cast<int64_t>(bytes / static_cast<std::streamoff>(sizeof(int64_t)));
    std::vector<int64_t> buffer(static_cast<size_t>(count));
    in.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(count * sizeof(int64_t)));
    if (!in)
    {
        throw std::runtime_error("Failed to read int64 tensor from " + path);
    }
    return torch::from_blob(buffer.data(), {count}, torch::TensorOptions().dtype(torch::kInt64)).clone();
}

} // namespace

int
main(int argc, char *argv[])
{
    RuntimeConfig config;
    CommandLine cmd(__FILE__);
    cmd.AddValue("data-dir", "Directory containing exported libtorch tensors", config.dataDir);
    cmd.AddValue("output-dir", "Directory for runtime outputs", config.outputDir);
    cmd.AddValue("init-state-dir", "Directory containing exported warm-start state tensors", config.initStateDir);
    cmd.AddValue("dataset", "Dataset name", config.dataset);
    cmd.AddValue("num-clients", "Number of federated clients", config.numClients);
    cmd.AddValue("num-planes", "Number of orbital planes", config.numPlanes);
    cmd.AddValue("rounds", "Federated rounds", config.rounds);
    cmd.AddValue("local-epochs", "Local epochs", config.localEpochs);
    cmd.AddValue("batch-size", "Batch size", config.batchSize);
    cmd.AddValue("input-dim", "Input feature count", config.inputDim);
    cmd.AddValue("seq-len", "Sequence length", config.seqLen);
    cmd.AddValue("num-classes", "Number of classes", config.numClasses);
    cmd.AddValue("train-samples", "Training sample count", config.trainSamples);
    cmd.AddValue("val-samples", "Validation sample count", config.valSamples);
    cmd.AddValue("test-samples", "Test sample count", config.testSamples);
    cmd.AddValue("hidden-dim", "GRU hidden size", config.hiddenDim);
    cmd.AddValue("conv-dim", "Front-end conv width", config.convDim);
    cmd.AddValue("dsc-dim", "DSC output width", config.dscDim);
    cmd.AddValue("fc-hidden", "Classifier hidden width", config.fcHidden);
    cmd.AddValue("dropout", "Classifier dropout", config.dropout);
    cmd.AddValue("beta-floor", "Lower bound for adaptive gossip mixing", config.betaFloor);
    cmd.AddValue("lambda-s", "Staleness decay", config.lambdaS);
    cmd.AddValue("rho", "Compensation factor", config.rho);
    cmd.AddValue("mu", "Reputation smoothing", config.mu);
    cmd.AddValue("global-momentum", "Blend factor for previous global model", config.globalMomentum);
    cmd.AddValue("lr", "Learning rate", config.lr);
    cmd.AddValue("weight-decay", "Weight decay", config.weightDecay);
    cmd.AddValue("device", "cpu or cuda", config.device);
    cmd.Parse(argc, argv);

    NS_ABORT_MSG_IF(config.dataDir.empty(), "data-dir must be provided");
    NS_ABORT_MSG_IF(config.trainSamples == 0 || config.valSamples == 0 || config.testSamples == 0,
                    "train/val/test sample counts must be provided");
    std::filesystem::create_directories(config.outputDir);

    torch::Tensor trainX, trainY, valX, valY, testX, testY;
    trainX = LoadFloatTensor(
        config.dataDir + "/train_X.f32",
        {static_cast<int64_t>(config.trainSamples), static_cast<int64_t>(config.seqLen), static_cast<int64_t>(config.inputDim)});
    trainY = LoadLongTensorFlat(config.dataDir + "/train_y.i64");
    valX = LoadFloatTensor(
        config.dataDir + "/val_X.f32",
        {static_cast<int64_t>(config.valSamples), static_cast<int64_t>(config.seqLen), static_cast<int64_t>(config.inputDim)});
    valY = LoadLongTensorFlat(config.dataDir + "/val_y.i64");
    testX = LoadFloatTensor(
        config.dataDir + "/test_X.f32",
        {static_cast<int64_t>(config.testSamples), static_cast<int64_t>(config.seqLen), static_cast<int64_t>(config.inputDim)});
    testY = LoadLongTensorFlat(config.dataDir + "/test_y.i64");

    torch::Device device = torch::kCPU;
    if (config.device == "cuda" && torch::cuda::is_available())
    {
        device = torch::Device(torch::kCUDA, 0);
    }

    std::vector<ClientState> clients;
    clients.reserve(config.numClients);
    for (uint32_t clientId = 0; clientId < config.numClients; ++clientId)
    {
        torch::Tensor indices;
        std::ostringstream path;
        path << config.dataDir << "/partitions/sat_" << clientId << ".i64";
        indices = LoadLongTensorFlat(path.str()).to(
            torch::TensorOptions().device(torch::kCPU).dtype(torch::kLong));
        clients.push_back(ClientState{clientId, clientId / (config.numClients / config.numPlanes), indices, 1.0, 0});
    }

    auto globalModel = CreateModel(config, device);
    if (!config.initStateDir.empty())
    {
        LoadStateDirIntoModel(config.initStateDir, *globalModel);
        std::cout << "Initialized 4B runtime from warm-start state: " << config.initStateDir << std::endl;
    }
    std::vector<std::shared_ptr<DscCbamGruImpl>> clientModels;
    std::vector<std::shared_ptr<DscCbamGruImpl>> planeModels;
    std::vector<std::shared_ptr<DscCbamGruImpl>> lastPlaneModels;
    std::vector<uint32_t> planeLastSync(config.numPlanes, 0);
    for (uint32_t i = 0; i < config.numClients; ++i)
    {
        clientModels.push_back(CloneModel(*globalModel, config, device));
    }
    for (uint32_t p = 0; p < config.numPlanes; ++p)
    {
        planeModels.push_back(CloneModel(*globalModel, config, device));
        lastPlaneModels.push_back(CloneModel(*globalModel, config, device));
    }

    std::mt19937 rng(config.seed);
    std::vector<std::map<std::string, double>> roundRows;
    std::vector<std::vector<double>> reputationHistory(config.numClients, std::vector<double>{1.0});
    EvalMetrics bestVal;
    EvalMetrics bestTest;
    uint32_t bestRound = 0;
    double bestValAcc = -1.0;
    const std::string bestModelPath = config.outputDir + "/best_global_model.pt";

    for (uint32_t roundIdx = 1; roundIdx <= config.rounds; ++roundIdx)
    {
        const bool fullMode = roundIdx > config.warmupRounds;
        const bool gossipEnabled = fullMode;
        auto previousGlobalModel = CloneModel(*globalModel, config, device);
        const double previousGlobalLoss = fullMode
                                              ? EvaluateModel(*previousGlobalModel, valX, valY, config.batchSize, config.numClasses, device).loss
                                              : 0.0;

        std::vector<std::vector<ClientUploadPayload>> planeUploads(config.numPlanes);
        uint32_t failedUploads = 0;
        uint32_t attemptedUploads = 0;
        uint32_t staleCount = 0;
        double communicationBytes = 0.0;

        for (auto &client : clients)
        {
            auto baseModel = CloneModel(*clientModels[client.clientId], config, device);
            auto localModel = CloneModel(*clientModels[client.clientId], config, device);
            const double localLoss = TrainOneClient(*localModel, trainX, trainY, client.indices, config, device);
            client.participationCount += 1;

            attemptedUploads += 1;
            auto intra = BuildLinkState(
                true,
                config.intraSuccessProb,
                config.intraDelayMs,
                config.intraBandwidthMbps,
                0.01,
                config.roundDurationSeconds,
                rng);
            if (!CanTransfer(*localModel, intra))
            {
                failedUploads += 1;
                continue;
            }

            const double staleness = std::max<int>(0, static_cast<int>(roundIdx) - static_cast<int>(client.lastSyncRound));
            staleCount += (staleness > 1.0) ? 1 : 0;
            const double weight = static_cast<double>(client.indices.size(0)) *
                                  std::exp(-config.lambdaS * staleness) *
                                  EstimateLinkQuality(intra) *
                                  client.reputation;
            planeUploads[client.planeId].push_back(ClientUploadPayload{
                client.clientId,
                client.planeId,
                localModel,
                baseModel,
                localLoss,
                client.indices.size(0),
                client.reputation,
                client.lastSyncRound,
                EstimateLinkQuality(intra),
            });
            clientModels[client.clientId] = localModel;
            client.lastSyncRound = roundIdx;
            communicationBytes += TensorBytes(*localModel);
        }

        for (uint32_t plane = 0; plane < config.numPlanes; ++plane)
        {
            lastPlaneModels[plane] = CloneModel(*planeModels[plane], config, device);
            if (!planeUploads[plane].empty())
            {
                std::vector<std::pair<std::shared_ptr<DscCbamGruImpl>, double>> weightedModels;
                for (const auto &payload : planeUploads[plane])
                {
                    const double staleness = std::max<int>(0, static_cast<int>(roundIdx) - static_cast<int>(payload.lastSyncRound));
                    double score = static_cast<double>(payload.sampleCount);
                    if (fullMode)
                    {
                        score *= std::exp(-config.lambdaS * staleness) * payload.linkQuality * payload.reputation;
                    }
                    weightedModels.push_back({payload.weights, score});
                }
                WeightedAverageInto(weightedModels, *planeModels[plane]);
                planeLastSync[plane] = roundIdx;
            }
        }

        std::vector<std::shared_ptr<DscCbamGruImpl>> gossipedPlaneModels;
        for (uint32_t plane = 0; plane < config.numPlanes; ++plane)
        {
            auto mixed = CloneModel(*planeModels[plane], config, device);
            if (gossipEnabled)
            {
                std::vector<std::pair<std::shared_ptr<DscCbamGruImpl>, double>> neighborModels;
                std::vector<double> neighborScores;
                for (uint32_t other = 0; other < config.numPlanes; ++other)
                {
                    if (other == plane)
                    {
                        continue;
                    }
                    if (!InterVisible(config, roundIdx, plane, other))
                    {
                        const double planeStaleness =
                            std::max<int>(0, static_cast<int>(roundIdx) - static_cast<int>(planeLastSync[other]));
                        auto compensated = CloneModel(*planeModels[plane], config, device);
                        WeightedAverageInto(
                            {
                                {CloneModel(*lastPlaneModels[other], config, device), config.rho},
                                {CloneModel(*planeModels[plane], config, device), 1.0 - config.rho},
                            },
                            *compensated);
                        neighborModels.push_back({compensated, 0.5 * std::exp(-config.lambdaS * planeStaleness)});
                        neighborScores.push_back(0.5 * std::exp(-config.lambdaS * planeStaleness));
                        continue;
                    }
                    auto inter = BuildLinkState(
                        true,
                        config.interSuccessProb,
                        config.interDelayMs,
                        config.interBandwidthMbps,
                        config.interLoss,
                        config.roundDurationSeconds,
                        rng);
                    if (CanTransfer(*planeModels[other], inter))
                    {
                        const double quality = EstimateLinkQuality(inter);
                        const double staleness =
                            std::max<int>(0, static_cast<int>(roundIdx) - static_cast<int>(planeLastSync[other]));
                        const double neighborScore = quality * std::exp(-config.lambdaS * staleness);
                        neighborModels.push_back({planeModels[other], neighborScore});
                        neighborScores.push_back(neighborScore);
                        communicationBytes += TensorBytes(*planeModels[other]);
                    }
                    else
                    {
                        const double staleness =
                            std::max<int>(0, static_cast<int>(roundIdx) - static_cast<int>(planeLastSync[other]));
                        auto compensated = CloneModel(*planeModels[plane], config, device);
                        WeightedAverageInto(
                            {
                                {CloneModel(*lastPlaneModels[other], config, device), config.rho},
                                {CloneModel(*planeModels[plane], config, device), 1.0 - config.rho},
                            },
                            *compensated);
                        neighborModels.push_back({compensated, 0.5 * std::exp(-config.lambdaS * staleness)});
                        neighborScores.push_back(0.5 * std::exp(-config.lambdaS * staleness));
                    }
                }
                if (!neighborModels.empty())
                {
                    auto neighborMix = CloneModel(*mixed, config, device);
                    WeightedAverageInto(neighborModels, *neighborMix);
                    const double avgNeighborQuality = neighborScores.empty()
                                                          ? 0.0
                                                          : std::accumulate(neighborScores.begin(), neighborScores.end(), 0.0) /
                                                                static_cast<double>(neighborScores.size());
                    const double adaptiveBeta =
                        config.betaFloor + (config.beta - config.betaFloor) * std::clamp(avgNeighborQuality, 0.0, 1.0);
                    WeightedAverageInto(
                        {
                            {CloneModel(*mixed, config, device), 1.0 - adaptiveBeta},
                            {neighborMix, adaptiveBeta},
                        },
                        *mixed);
                }
            }
            gossipedPlaneModels.push_back(mixed);
        }

        std::vector<std::pair<std::shared_ptr<DscCbamGruImpl>, double>> globalSources;
        for (uint32_t plane = 0; plane < config.numPlanes; ++plane)
        {
            double planeWeight = 1.0;
            for (const auto &entry : planeUploads[plane])
            {
                planeWeight += static_cast<double>(entry.sampleCount);
            }
            globalSources.push_back({gossipedPlaneModels[plane], planeWeight});
            planeModels[plane] = gossipedPlaneModels[plane];
        }
        auto mergedGlobal = CloneModel(*globalModel, config, device);
        WeightedAverageInto(globalSources, *mergedGlobal);
        if (fullMode)
        {
            WeightedAverageInto(
                {
                    {previousGlobalModel, config.globalMomentum},
                    {mergedGlobal, 1.0 - config.globalMomentum},
                },
                *globalModel);
        }
        else
        {
            CopyState(*mergedGlobal, *globalModel);
        }

        for (auto &client : clients)
        {
            clientModels[client.clientId] = CloneModel(*planeModels[client.planeId], config, device);
            client.successfulSyncCount += 1;
        }

        if (fullMode)
        {
            for (uint32_t plane = 0; plane < config.numPlanes; ++plane)
            {
                for (const auto &payload : planeUploads[plane])
                {
                    auto &client = clients[payload.clientId];
                    const double candidateLoss =
                        EvaluateModel(*payload.weights, valX, valY, config.batchSize, config.numClasses, device).loss;
                    const double sim =
                        (CosineSimilarityDelta(*payload.weights, *payload.baseModel, *planeModels[plane], *previousGlobalModel) + 1.0) / 2.0;
                    const double improve = BoundedImprovement(previousGlobalLoss, candidateLoss);
                    const double stable = static_cast<double>(client.successfulSyncCount) /
                                          static_cast<double>(std::max<uint32_t>(1, client.participationCount));
                    const double score =
                        (config.scoreSimWeight * sim) +
                        (config.scoreImproveWeight * improve) +
                        (config.scoreStableWeight * std::clamp(stable, 0.0, 1.0));
                    client.reputation = std::clamp(
                        (config.mu * client.reputation) + ((1.0 - config.mu) * score),
                        config.rMin,
                        1.0);
                }
            }
        }
        for (const auto &client : clients)
        {
            reputationHistory[client.clientId].push_back(client.reputation);
        }

        auto valMetrics = EvaluateModel(*globalModel, valX, valY, config.batchSize, config.numClasses, device);
        auto testMetrics = EvaluateModel(*globalModel, testX, testY, config.batchSize, config.numClasses, device);

        const double failureRatio = attemptedUploads > 0 ? static_cast<double>(failedUploads) / attemptedUploads : 0.0;
        const double staleRatio = attemptedUploads > 0 ? static_cast<double>(staleCount) / attemptedUploads : 0.0;
        roundRows.push_back({
            {"round", static_cast<double>(roundIdx)},
            {"val_loss", valMetrics.loss},
            {"val_accuracy", valMetrics.accuracy},
            {"val_precision", valMetrics.precision},
            {"val_recall", valMetrics.recall},
            {"val_f1", valMetrics.f1},
            {"test_accuracy", testMetrics.accuracy},
            {"test_precision", testMetrics.precision},
            {"test_recall", testMetrics.recall},
            {"test_f1", testMetrics.f1},
            {"communication_cost_mb", communicationBytes / (1024.0 * 1024.0)},
            {"stale_update_ratio", staleRatio},
            {"link_failure_robustness", 1.0 - failureRatio},
        });

        std::cout << "Round " << roundIdx
                  << " | val_acc=" << std::fixed << std::setprecision(4) << valMetrics.accuracy
                  << " | test_acc=" << testMetrics.accuracy
                  << " | comm_mb=" << (communicationBytes / (1024.0 * 1024.0))
                  << std::endl;

        if (valMetrics.accuracy > bestValAcc)
        {
            bestValAcc = valMetrics.accuracy;
            bestVal = valMetrics;
            bestTest = testMetrics;
            bestRound = roundIdx;
            torch::serialize::OutputArchive archive;
            globalModel->save(archive);
            archive.save_to(bestModelPath);
        }
    }

    WriteRoundCsv(config.outputDir + "/round_metrics.csv", roundRows);
    WriteSummary(config.outputDir + "/summary.json", config, bestVal, bestTest, bestRound, bestModelPath);
    WriteReputationHistory(config.outputDir + "/reputation_history.json", reputationHistory);
    return 0;
}
