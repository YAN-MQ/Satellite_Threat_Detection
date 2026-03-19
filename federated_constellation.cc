/*
 * Federated constellation communication-environment simulator for OrbitShield_FL
 *
 * This program is intentionally independent from realtime_satellite.cc.
 * It does not generate traffic datasets. Instead, it simulates the round-level
 * communication environment required by the federated training pipeline.
 *
 * Output artifacts:
 *   - constellation_config.json
 *   - manifest.json
 *   - round_0001.json ... round_N.json
 *
 * The exported traces include:
 *   - plane membership
 *   - static satellite links
 *   - intra-plane aggregated link state
 *   - inter-plane aggregated link state
 *   - per-satellite link state for future extension
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace ns3;

namespace
{

struct SimulationConfig
{
    uint32_t numPlanes = 3;
    uint32_t satsPerPlane = 4;
    uint32_t rounds = 20;
    double roundDurationSeconds = 30.0;

    std::string intraRate = "500Mbps";
    std::string intraDelay = "10ms";
    double intraLoss = 0.01;
    double intraSuccessProb = 0.98;

    std::string interRate = "120Mbps";
    std::string interDelay = "25ms";
    double interLoss = 0.05;
    double interSuccessProb = 0.75;

    uint32_t contactPeriod = 4;
    uint32_t contactDurationRounds = 2;
    std::string interPlaneMode = "ring";

    std::string outputDir = "trace_ns3/default";
    uint32_t seed = 42;
};

struct SatelliteInfo
{
    std::string satelliteId;
    uint32_t globalIndex = 0;
    uint32_t planeId = 0;
    uint32_t localIndex = 0;
};

struct StaticLinkInfo
{
    std::string linkId;
    std::string linkType;
    std::string endpointA;
    std::string endpointB;
    uint32_t planeA = 0;
    uint32_t planeB = 0;
    std::string subnet;
    std::string nominalRate;
    std::string nominalDelay;
    double nominalLoss = 0.0;
};

struct RuntimeLinkState
{
    bool available = false;
    bool success = false;
    double delayMs = 0.0;
    double bandwidthMbps = 0.0;
    double packetLoss = 1.0;
    double contactDurationSeconds = 0.0;
};

std::string
Quote(const std::string &value)
{
    std::ostringstream oss;
    oss << '"';
    for (const char c : value)
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
        case '\r':
            oss << "\\r";
            break;
        case '\t':
            oss << "\\t";
            break;
        default:
            oss << c;
            break;
        }
    }
    oss << '"';
    return oss.str();
}

void
Indent(std::ostream &os, uint32_t level)
{
    for (uint32_t i = 0; i < level; ++i)
    {
        os << "  ";
    }
}

std::string
ToSatId(uint32_t index)
{
    std::ostringstream oss;
    oss << "sat_" << index;
    return oss.str();
}

std::string
ToPlaneKey(uint32_t planeId)
{
    return std::to_string(planeId);
}

double
ParseRateMbps(const std::string &rate)
{
    const auto suffixPos = rate.find("Mbps");
    if (suffixPos != std::string::npos)
    {
        return std::stod(rate.substr(0, suffixPos));
    }
    return std::stod(rate);
}

double
ParseDelayMs(const std::string &delay)
{
    const auto suffixPos = delay.find("ms");
    if (suffixPos != std::string::npos)
    {
        return std::stod(delay.substr(0, suffixPos));
    }
    return std::stod(delay);
}

std::string
MakeSubnet(uint32_t subnetIndex)
{
    const uint32_t octet2 = 1 + ((subnetIndex / 250) % 250);
    const uint32_t octet3 = 1 + (subnetIndex % 250);
    std::ostringstream oss;
    oss << "10." << octet2 << "." << octet3 << ".0";
    return oss.str();
}

RuntimeLinkState
BuildLinkState(
    bool available,
    double nominalDelayMs,
    double nominalBandwidthMbps,
    double nominalLoss,
    double successProbability,
    double contactDurationSeconds,
    Ptr<UniformRandomVariable> jitterRv,
    Ptr<UniformRandomVariable> successRv)
{
    RuntimeLinkState state;
    state.available = available;
    if (!available)
    {
        state.success = false;
        state.delayMs = 0.0;
        state.bandwidthMbps = 0.0;
        state.packetLoss = 1.0;
        state.contactDurationSeconds = 0.0;
        return state;
    }

    const double delayJitter = jitterRv->GetValue(0.90, 1.10);
    const double bandwidthJitter = jitterRv->GetValue(0.85, 1.15);
    const double lossJitter = jitterRv->GetValue(0.80, 1.20);

    state.success = successRv->GetValue(0.0, 1.0) < successProbability;
    state.delayMs = std::max(0.1, nominalDelayMs * delayJitter);
    state.bandwidthMbps = std::max(0.1, nominalBandwidthMbps * bandwidthJitter);
    state.packetLoss = std::clamp(nominalLoss * lossJitter, 0.0, 1.0);
    state.contactDurationSeconds = contactDurationSeconds;
    return state;
}

class FederatedConstellationExporter
{
  public:
    explicit FederatedConstellationExporter(const SimulationConfig &config)
        : m_config(config),
          m_intraNominalRateMbps(ParseRateMbps(config.intraRate)),
          m_intraNominalDelayMs(ParseDelayMs(config.intraDelay)),
          m_interNominalRateMbps(ParseRateMbps(config.interRate)),
          m_interNominalDelayMs(ParseDelayMs(config.interDelay))
    {
        std::filesystem::create_directories(m_config.outputDir);

        RngSeedManager::SetSeed(m_config.seed);
        RngSeedManager::SetRun(1);

        m_jitterRv = CreateObject<UniformRandomVariable>();
        m_successRv = CreateObject<UniformRandomVariable>();

        BuildSatellites();
        BuildNs3Topology();
    }

    void Run()
    {
        WriteConstellationConfig();
        for (uint32_t round = 1; round <= m_config.rounds; ++round)
        {
            Simulator::Schedule(
                Seconds((round - 1) * m_config.roundDurationSeconds),
                &FederatedConstellationExporter::ExportRoundTrace,
                this,
                round);
        }

        Simulator::Stop(Seconds(m_config.rounds * m_config.roundDurationSeconds + 0.001));
        Simulator::Run();
        Simulator::Destroy();

        WriteManifest();
    }

  private:
    void BuildSatellites()
    {
        const uint32_t totalSats = m_config.numPlanes * m_config.satsPerPlane;
        m_nodes.Create(totalSats);

        InternetStackHelper stack;
        stack.Install(m_nodes);

        for (uint32_t plane = 0; plane < m_config.numPlanes; ++plane)
        {
            std::vector<SatelliteInfo> planeMembers;
            for (uint32_t local = 0; local < m_config.satsPerPlane; ++local)
            {
                const uint32_t globalIndex = plane * m_config.satsPerPlane + local;
                SatelliteInfo info;
                info.satelliteId = ToSatId(globalIndex);
                info.globalIndex = globalIndex;
                info.planeId = plane;
                info.localIndex = local;
                planeMembers.push_back(info);
                m_satellites.push_back(info);
            }
            m_planes.emplace(plane, planeMembers);
        }
    }

    void BuildNs3Topology()
    {
        PointToPointHelper intraP2p;
        intraP2p.SetDeviceAttribute("DataRate", StringValue(m_config.intraRate));
        intraP2p.SetChannelAttribute("Delay", StringValue(m_config.intraDelay));

        PointToPointHelper interP2p;
        interP2p.SetDeviceAttribute("DataRate", StringValue(m_config.interRate));
        interP2p.SetChannelAttribute("Delay", StringValue(m_config.interDelay));

        Ipv4AddressHelper address;
        uint32_t subnetIndex = 0;
        std::set<std::pair<uint32_t, uint32_t>> seenIntraPairs;

        for (uint32_t plane = 0; plane < m_config.numPlanes; ++plane)
        {
            const auto &members = m_planes.at(plane);
            if (members.size() < 2)
            {
                continue;
            }
            for (uint32_t local = 0; local < members.size(); ++local)
            {
                const uint32_t nextLocal = (local + 1) % members.size();
                const auto logicalPair = std::make_pair(std::min(local, nextLocal), std::max(local, nextLocal));
                if (!seenIntraPairs.insert(std::make_pair(plane * 1000 + logicalPair.first, plane * 1000 + logicalPair.second)).second)
                {
                    continue;
                }

                NodeContainer linkNodes(m_nodes.Get(members[local].globalIndex), m_nodes.Get(members[nextLocal].globalIndex));
                NetDeviceContainer devices = intraP2p.Install(linkNodes);

                const std::string subnet = MakeSubnet(subnetIndex++);
                address.SetBase(Ipv4Address(subnet.c_str()), "255.255.255.0");
                address.Assign(devices);

                StaticLinkInfo link;
                link.linkId = "intra_p" + std::to_string(plane) + "_" + std::to_string(local) + "_" + std::to_string(nextLocal);
                link.linkType = "intra_plane";
                link.endpointA = members[local].satelliteId;
                link.endpointB = members[nextLocal].satelliteId;
                link.planeA = plane;
                link.planeB = plane;
                link.subnet = subnet;
                link.nominalRate = m_config.intraRate;
                link.nominalDelay = m_config.intraDelay;
                link.nominalLoss = m_config.intraLoss;
                m_staticLinks.push_back(link);
            }
        }

        if (m_config.interPlaneMode != "ring")
        {
            NS_FATAL_ERROR("Only inter-plane mode 'ring' is currently supported.");
        }

        std::set<std::pair<uint32_t, uint32_t>> seenInterPairs;
        for (uint32_t plane = 0; plane < m_config.numPlanes; ++plane)
        {
            const uint32_t nextPlane = (plane + 1) % m_config.numPlanes;
            const auto planePair = std::make_pair(std::min(plane, nextPlane), std::max(plane, nextPlane));
            if (!seenInterPairs.insert(planePair).second)
            {
                continue;
            }

            const auto &leftMembers = m_planes.at(plane);
            const auto &rightMembers = m_planes.at(nextPlane);
            for (uint32_t local = 0; local < m_config.satsPerPlane; ++local)
            {
                NodeContainer linkNodes(m_nodes.Get(leftMembers[local].globalIndex), m_nodes.Get(rightMembers[local].globalIndex));
                NetDeviceContainer devices = interP2p.Install(linkNodes);

                const std::string subnet = MakeSubnet(subnetIndex++);
                address.SetBase(Ipv4Address(subnet.c_str()), "255.255.255.0");
                address.Assign(devices);

                StaticLinkInfo link;
                link.linkId = "inter_p" + std::to_string(plane) + "_p" + std::to_string(nextPlane) + "_l" + std::to_string(local);
                link.linkType = "inter_plane";
                link.endpointA = leftMembers[local].satelliteId;
                link.endpointB = rightMembers[local].satelliteId;
                link.planeA = plane;
                link.planeB = nextPlane;
                link.subnet = subnet;
                link.nominalRate = m_config.interRate;
                link.nominalDelay = m_config.interDelay;
                link.nominalLoss = m_config.interLoss;
                m_staticLinks.push_back(link);
            }
        }
    }

    std::map<uint32_t, RuntimeLinkState> BuildIntraPlaneStates() const
    {
        std::map<uint32_t, RuntimeLinkState> states;
        for (uint32_t plane = 0; plane < m_config.numPlanes; ++plane)
        {
            states.emplace(
                plane,
                BuildLinkState(
                    true,
                    m_intraNominalDelayMs,
                    m_intraNominalRateMbps,
                    m_config.intraLoss,
                    m_config.intraSuccessProb,
                    m_config.roundDurationSeconds,
                    m_jitterRv,
                    m_successRv));
        }
        return states;
    }

    std::map<std::pair<uint32_t, uint32_t>, RuntimeLinkState> BuildInterPlaneStates(uint32_t round) const
    {
        std::map<std::pair<uint32_t, uint32_t>, RuntimeLinkState> states;
        for (uint32_t plane = 0; plane < m_config.numPlanes; ++plane)
        {
            const uint32_t nextPlane = (plane + 1) % m_config.numPlanes;
            const auto pair = std::make_pair(std::min(plane, nextPlane), std::max(plane, nextPlane));
            const uint32_t phaseOffset = pair.first;
            const bool visible =
                ((round - 1 + phaseOffset) % m_config.contactPeriod) < m_config.contactDurationRounds;
            states.emplace(
                pair,
                BuildLinkState(
                    visible,
                    m_interNominalDelayMs,
                    m_interNominalRateMbps,
                    m_config.interLoss,
                    m_config.interSuccessProb,
                    visible ? m_config.roundDurationSeconds : 0.0,
                    m_jitterRv,
                    m_successRv));
        }
        return states;
    }

    std::string BuildRoundFileName(uint32_t round) const
    {
        std::ostringstream oss;
        oss << "round_" << std::setw(4) << std::setfill('0') << round << ".json";
        return oss.str();
    }

    void ExportRoundTrace(uint32_t round)
    {
        const auto intraStates = BuildIntraPlaneStates();
        const auto interStates = BuildInterPlaneStates(round);
        const std::string fileName = BuildRoundFileName(round);
        const std::filesystem::path outputPath = std::filesystem::path(m_config.outputDir) / fileName;
        m_roundFiles.push_back(fileName);

        std::ofstream out(outputPath);
        out << std::fixed << std::setprecision(6);
        out << "{\n";
        Indent(out, 1);
        out << "\"schema_version\": 1,\n";
        Indent(out, 1);
        out << "\"round\": " << round << ",\n";
        Indent(out, 1);
        out << "\"sim_time_s\": " << Simulator::Now().GetSeconds() << ",\n";
        Indent(out, 1);
        out << "\"round_duration_s\": " << m_config.roundDurationSeconds << ",\n";
        Indent(out, 1);
        out << "\"num_planes\": " << m_config.numPlanes << ",\n";
        Indent(out, 1);
        out << "\"sats_per_plane\": " << m_config.satsPerPlane << ",\n";

        Indent(out, 1);
        out << "\"planes\": {\n";
        for (uint32_t plane = 0; plane < m_config.numPlanes; ++plane)
        {
            Indent(out, 2);
            out << Quote(ToPlaneKey(plane)) << ": [";
            const auto &members = m_planes.at(plane);
            for (size_t i = 0; i < members.size(); ++i)
            {
                out << Quote(members[i].satelliteId);
                if (i + 1 != members.size())
                {
                    out << ", ";
                }
            }
            out << "]";
            if (plane + 1 != m_config.numPlanes)
            {
                out << ",";
            }
            out << "\n";
        }
        Indent(out, 1);
        out << "},\n";

        Indent(out, 1);
        out << "\"satellites\": {\n";
        for (size_t i = 0; i < m_satellites.size(); ++i)
        {
            const auto &sat = m_satellites[i];
            Indent(out, 2);
            out << Quote(sat.satelliteId) << ": {\"plane_id\": " << sat.planeId << ", \"local_index\": " << sat.localIndex
                << "}";
            if (i + 1 != m_satellites.size())
            {
                out << ",";
            }
            out << "\n";
        }
        Indent(out, 1);
        out << "},\n";

        Indent(out, 1);
        out << "\"intra_plane_links\": {\n";
        for (uint32_t plane = 0; plane < m_config.numPlanes; ++plane)
        {
            const auto &state = intraStates.at(plane);
            Indent(out, 2);
            out << Quote(ToPlaneKey(plane)) << ": ";
            WriteRuntimeState(out, state);
            if (plane + 1 != m_config.numPlanes)
            {
                out << ",";
            }
            out << "\n";
        }
        Indent(out, 1);
        out << "},\n";

        Indent(out, 1);
        out << "\"inter_plane_links\": {\n";
        const auto interEntries = SortedInterEntries(interStates);
        for (size_t i = 0; i < interEntries.size(); ++i)
        {
            const auto &[pair, state] = interEntries[i];
            Indent(out, 2);
            out << Quote(std::to_string(pair.first) + "-" + std::to_string(pair.second)) << ": ";
            WriteRuntimeState(out, state);
            if (i + 1 != interEntries.size())
            {
                out << ",";
            }
            out << "\n";
        }
        Indent(out, 1);
        out << "},\n";

        Indent(out, 1);
        out << "\"satellite_links\": [\n";
        for (size_t i = 0; i < m_staticLinks.size(); ++i)
        {
            const auto &link = m_staticLinks[i];
            const RuntimeLinkState state = ResolveStaticLinkState(link, intraStates, interStates);
            Indent(out, 2);
            out << "{";
            out << "\"link_id\": " << Quote(link.linkId) << ", ";
            out << "\"link_type\": " << Quote(link.linkType) << ", ";
            out << "\"endpoint_a\": " << Quote(link.endpointA) << ", ";
            out << "\"endpoint_b\": " << Quote(link.endpointB) << ", ";
            out << "\"plane_a\": " << link.planeA << ", ";
            out << "\"plane_b\": " << link.planeB << ", ";
            out << "\"subnet\": " << Quote(link.subnet) << ", ";
            out << "\"available\": " << (state.available ? "true" : "false") << ", ";
            out << "\"success\": " << (state.success ? "true" : "false") << ", ";
            out << "\"delay_ms\": " << state.delayMs << ", ";
            out << "\"bandwidth_mbps\": " << state.bandwidthMbps << ", ";
            out << "\"packet_loss\": " << state.packetLoss << ", ";
            out << "\"contact_duration_s\": " << state.contactDurationSeconds;
            out << "}";
            if (i + 1 != m_staticLinks.size())
            {
                out << ",";
            }
            out << "\n";
        }
        Indent(out, 1);
        out << "]\n";
        out << "}\n";
    }

    static void WriteRuntimeState(std::ostream &out, const RuntimeLinkState &state)
    {
        out << "{";
        out << "\"available\": " << (state.available ? "true" : "false") << ", ";
        out << "\"success\": " << (state.success ? "true" : "false") << ", ";
        out << "\"delay_ms\": " << state.delayMs << ", ";
        out << "\"bandwidth_mbps\": " << state.bandwidthMbps << ", ";
        out << "\"packet_loss\": " << state.packetLoss << ", ";
        out << "\"contact_duration_s\": " << state.contactDurationSeconds;
        out << "}";
    }

    static std::vector<std::pair<std::pair<uint32_t, uint32_t>, RuntimeLinkState>>
    SortedInterEntries(const std::map<std::pair<uint32_t, uint32_t>, RuntimeLinkState> &states)
    {
        std::vector<std::pair<std::pair<uint32_t, uint32_t>, RuntimeLinkState>> entries;
        for (const auto &entry : states)
        {
            entries.push_back(entry);
        }
        std::sort(entries.begin(), entries.end(), [](const auto &left, const auto &right) {
            return left.first < right.first;
        });
        return entries;
    }

    static RuntimeLinkState ResolveStaticLinkState(
        const StaticLinkInfo &link,
        const std::map<uint32_t, RuntimeLinkState> &intraStates,
        const std::map<std::pair<uint32_t, uint32_t>, RuntimeLinkState> &interStates)
    {
        if (link.linkType == "intra_plane")
        {
            return intraStates.at(link.planeA);
        }
        const auto pair = std::make_pair(std::min(link.planeA, link.planeB), std::max(link.planeA, link.planeB));
        return interStates.at(pair);
    }

    void WriteConstellationConfig() const
    {
        const std::filesystem::path outputPath = std::filesystem::path(m_config.outputDir) / "constellation_config.json";
        std::ofstream out(outputPath);
        out << std::fixed << std::setprecision(6);
        out << "{\n";
        Indent(out, 1);
        out << "\"schema_version\": 1,\n";
        Indent(out, 1);
        out << "\"generator\": " << Quote("federated_constellation.cc") << ",\n";
        Indent(out, 1);
        out << "\"num_planes\": " << m_config.numPlanes << ",\n";
        Indent(out, 1);
        out << "\"sats_per_plane\": " << m_config.satsPerPlane << ",\n";
        Indent(out, 1);
        out << "\"total_satellites\": " << (m_config.numPlanes * m_config.satsPerPlane) << ",\n";
        Indent(out, 1);
        out << "\"rounds\": " << m_config.rounds << ",\n";
        Indent(out, 1);
        out << "\"round_duration_s\": " << m_config.roundDurationSeconds << ",\n";
        Indent(out, 1);
        out << "\"seed\": " << m_config.seed << ",\n";
        Indent(out, 1);
        out << "\"inter_plane_mode\": " << Quote(m_config.interPlaneMode) << ",\n";
        Indent(out, 1);
        out << "\"intra_plane\": {\"rate\": " << Quote(m_config.intraRate) << ", \"delay\": " << Quote(m_config.intraDelay)
            << ", \"loss\": " << m_config.intraLoss << ", \"success_prob\": " << m_config.intraSuccessProb << "},\n";
        Indent(out, 1);
        out << "\"inter_plane\": {\"rate\": " << Quote(m_config.interRate) << ", \"delay\": " << Quote(m_config.interDelay)
            << ", \"loss\": " << m_config.interLoss << ", \"success_prob\": " << m_config.interSuccessProb
            << ", \"contact_period\": " << m_config.contactPeriod
            << ", \"contact_duration_rounds\": " << m_config.contactDurationRounds << "},\n";

        Indent(out, 1);
        out << "\"planes\": {\n";
        for (uint32_t plane = 0; plane < m_config.numPlanes; ++plane)
        {
            Indent(out, 2);
            out << Quote(ToPlaneKey(plane)) << ": [";
            const auto &members = m_planes.at(plane);
            for (size_t i = 0; i < members.size(); ++i)
            {
                out << Quote(members[i].satelliteId);
                if (i + 1 != members.size())
                {
                    out << ", ";
                }
            }
            out << "]";
            if (plane + 1 != m_config.numPlanes)
            {
                out << ",";
            }
            out << "\n";
        }
        Indent(out, 1);
        out << "},\n";

        Indent(out, 1);
        out << "\"static_links\": [\n";
        for (size_t i = 0; i < m_staticLinks.size(); ++i)
        {
            const auto &link = m_staticLinks[i];
            Indent(out, 2);
            out << "{";
            out << "\"link_id\": " << Quote(link.linkId) << ", ";
            out << "\"link_type\": " << Quote(link.linkType) << ", ";
            out << "\"endpoint_a\": " << Quote(link.endpointA) << ", ";
            out << "\"endpoint_b\": " << Quote(link.endpointB) << ", ";
            out << "\"plane_a\": " << link.planeA << ", ";
            out << "\"plane_b\": " << link.planeB << ", ";
            out << "\"subnet\": " << Quote(link.subnet) << ", ";
            out << "\"nominal_rate\": " << Quote(link.nominalRate) << ", ";
            out << "\"nominal_delay\": " << Quote(link.nominalDelay) << ", ";
            out << "\"nominal_loss\": " << link.nominalLoss;
            out << "}";
            if (i + 1 != m_staticLinks.size())
            {
                out << ",";
            }
            out << "\n";
        }
        Indent(out, 1);
        out << "]\n";
        out << "}\n";
    }

    void WriteManifest() const
    {
        const std::filesystem::path outputPath = std::filesystem::path(m_config.outputDir) / "manifest.json";
        std::ofstream out(outputPath);
        out << "{\n";
        Indent(out, 1);
        out << "\"schema_version\": 1,\n";
        Indent(out, 1);
        out << "\"generator\": " << Quote("federated_constellation.cc") << ",\n";
        Indent(out, 1);
        out << "\"config_file\": " << Quote("constellation_config.json") << ",\n";
        Indent(out, 1);
        out << "\"round_count\": " << m_roundFiles.size() << ",\n";
        Indent(out, 1);
        out << "\"round_files\": [\n";
        for (size_t i = 0; i < m_roundFiles.size(); ++i)
        {
            Indent(out, 2);
            out << Quote(m_roundFiles[i]);
            if (i + 1 != m_roundFiles.size())
            {
                out << ",";
            }
            out << "\n";
        }
        Indent(out, 1);
        out << "]\n";
        out << "}\n";
    }

  private:
    SimulationConfig m_config;
    NodeContainer m_nodes;
    std::vector<SatelliteInfo> m_satellites;
    std::map<uint32_t, std::vector<SatelliteInfo>> m_planes;
    std::vector<StaticLinkInfo> m_staticLinks;
    std::vector<std::string> m_roundFiles;

    double m_intraNominalRateMbps;
    double m_intraNominalDelayMs;
    double m_interNominalRateMbps;
    double m_interNominalDelayMs;

    Ptr<UniformRandomVariable> m_jitterRv;
    Ptr<UniformRandomVariable> m_successRv;
};

} // namespace

int
main(int argc, char *argv[])
{
    SimulationConfig config;

    CommandLine cmd(__FILE__);
    cmd.AddValue("num-planes", "Number of orbital planes", config.numPlanes);
    cmd.AddValue("sats-per-plane", "Number of satellites in each orbital plane", config.satsPerPlane);
    cmd.AddValue("rounds", "Number of federated rounds to export", config.rounds);
    cmd.AddValue("round-duration", "Duration of one federated round in seconds", config.roundDurationSeconds);

    cmd.AddValue("intra-rate", "Nominal intra-plane data rate, e.g. 500Mbps", config.intraRate);
    cmd.AddValue("intra-delay", "Nominal intra-plane delay, e.g. 10ms", config.intraDelay);
    cmd.AddValue("intra-loss", "Nominal intra-plane packet loss probability", config.intraLoss);
    cmd.AddValue("intra-success-prob", "Probability that an intra-plane round communication succeeds", config.intraSuccessProb);

    cmd.AddValue("inter-rate", "Nominal inter-plane data rate, e.g. 120Mbps", config.interRate);
    cmd.AddValue("inter-delay", "Nominal inter-plane delay, e.g. 25ms", config.interDelay);
    cmd.AddValue("inter-loss", "Nominal inter-plane packet loss probability", config.interLoss);
    cmd.AddValue("inter-success-prob", "Probability that a visible inter-plane round communication succeeds", config.interSuccessProb);

    cmd.AddValue("contact-period", "Inter-plane contact visibility period measured in rounds", config.contactPeriod);
    cmd.AddValue("contact-duration-rounds", "Number of visible rounds in each contact period", config.contactDurationRounds);
    cmd.AddValue("inter-plane-mode", "Inter-plane topology mode, currently only 'ring'", config.interPlaneMode);

    cmd.AddValue("output-dir", "Directory used to store constellation traces", config.outputDir);
    cmd.AddValue("seed", "Seed used for ns-3 random variables", config.seed);
    cmd.Parse(argc, argv);

    if (config.numPlanes == 0 || config.satsPerPlane == 0 || config.rounds == 0)
    {
        NS_FATAL_ERROR("num-planes, sats-per-plane, and rounds must all be greater than zero.");
    }
    if (config.contactPeriod == 0 || config.contactDurationRounds == 0)
    {
        NS_FATAL_ERROR("contact-period and contact-duration-rounds must both be greater than zero.");
    }
    if (config.contactDurationRounds > config.contactPeriod)
    {
        NS_FATAL_ERROR("contact-duration-rounds cannot exceed contact-period.");
    }

    std::cout << "=== OrbitShield_FL Federated Constellation Simulator ===" << std::endl;
    std::cout << "Planes: " << config.numPlanes << ", satellites/plane: " << config.satsPerPlane << std::endl;
    std::cout << "Rounds: " << config.rounds << ", round duration: " << config.roundDurationSeconds << " s" << std::endl;
    std::cout << "Output dir: " << config.outputDir << std::endl;

    FederatedConstellationExporter exporter(config);
    exporter.Run();

    std::cout << "Constellation trace export completed." << std::endl;
    return 0;
}
