"""Quick check of delivered_photons"""

from multi_region_network import MultiRegionNetwork

network = MultiRegionNetwork()
network.add_region(region_id=0, n_synapses=2, name="Source")
network.add_region(region_id=1, n_synapses=2, name="Target")
network.connect_regions(0, 1, distance_mm=1.0)

dt = 0.001
active_stim = {'voltage': -10e-3, 'reward': False}

print("Checking delivered_photons values...")
print("-" * 50)

for step in range(200):
    network.time += dt
    
    for region_id, region in network.regions.items():
        region.step(dt, active_stim)
    
    for region_id, region in network.regions.items():
        packets = region.get_emitted_packets()
        if packets:
            network.waveguide_network.inject_packets(
                source_id=region_id,
                packets=packets,
                current_time=network.time
            )
    
    deliveries = network.waveguide_network.step(network.time)
    
    for target_id, packet_records in deliveries.items():
        for record in packet_records:
            # Show full record structure
            print(f"Step {step}: Record keys = {record.keys()}")
            print(f"  delivered_photons = {record.get('delivered_photons', 'NOT FOUND')}")
            print(f"  packet present = {'packet' in record}")
            if 'packet' in record:
                p = record['packet']
                print(f"  packet.n_photons = {p.n_photons}")
            print()
            
            # Only show first few
            if step > 10:
                print("... (stopping after 10 examples)")
                exit()