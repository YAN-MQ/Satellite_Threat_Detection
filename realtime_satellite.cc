/*
 * Module 2: NS-3 Real-time Satellite Channel Emulation
 * Uses TapBridge to connect with Linux virtual interfaces
 * Simulates LEO satellite link with 300Mbps bandwidth and 25ms delay
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/tap-bridge-module.h"

using namespace ns3;

int main(int argc, char *argv[])
{
    std::string tapLeft = "tap-left";
    std::string tapRight = "tap-right";
    double simulationTime = 60.0;
    
    CommandLine cmd;
    cmd.AddValue("tap-left", "Left tap interface", tapLeft);
    cmd.AddValue("tap-right", "Right tap interface", tapRight);
    cmd.AddValue("time", "Simulation time", simulationTime);
    cmd.Parse(argc, argv);
    
    // Enable realtime simulator for real-time emulation
    GlobalValue::Bind("SimulatorImplementationType", StringValue("ns3::RealtimeSimulatorImpl"));
    GlobalValue::Bind("ChecksumEnabled", BooleanValue(true));
    
    NS_LOG_INFO("=== LEO Satellite Channel Emulation ===");
    NS_LOG_INFO("Tap devices: " << tapLeft << ", " << tapRight);
    
    // Create two nodes: Sender and Receiver
    NodeContainer nodes;
    nodes.Create(2);
    
    // Point-to-point channel (satellite link)
    // Bandwidth: 300Mbps, Delay: 25ms
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("300Mbps"));
    p2p.SetChannelAttribute("Delay", StringValue("25ms"));
    
    NetDeviceContainer devices = p2p.Install(nodes);
    
    // Install internet stack
    InternetStackHelper stack;
    stack.Install(nodes);
    
    // Assign IP addresses
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    address.Assign(devices);
    
    // Setup TapBridge for sender (Node 0) - receives traffic from tcpreplay
    TapBridgeHelper tapBridgeLeft;
    tapBridgeLeft.SetAttribute("Mode", StringValue("UseLocal"));
    tapBridgeLeft.SetAttribute("DeviceName", StringValue(tapLeft));
    tapBridgeLeft.Install(nodes.Get(0), devices.Get(0));
    
    // Setup TapBridge for receiver (Node 1) - captures output traffic
    TapBridgeHelper tapBridgeRight;
    tapBridgeRight.SetAttribute("Mode", StringValue("UseLocal"));
    tapBridgeRight.SetAttribute("DeviceName", StringValue(tapRight));
    tapBridgeRight.Install(nodes.Get(1), devices.Get(1));
    
    std::cout << "NS-3 realtime simulation ready." << std::endl;
    std::cout << "Tap devices: " << tapLeft << ", " << tapRight << std::endl;
    
    // Run simulation
    Simulator::Stop(Seconds(simulationTime));
    Simulator::Run();
    Simulator::Destroy();
    
    std::cout << "Simulation completed." << std::endl;
    return 0;
}
