"""
Debug Cross-Neuron Entanglement
===============================

Traces the full pathway to find where bonds fail to form:
1. Are photons being emitted?
2. Are they being tagged with dimer_link?
3. Are they arriving at target?
4. Is process_arrival being called?
5. Are conditions for bond formation met?
"""

from multi_region_network import MultiRegionNetwork
from cross_neuron_entanglement import CrossNeuronEntanglementTracker

def debug_entanglement():
    print("=" * 70)
    print("DEBUG: Cross-Neuron Entanglement Chain")
    print("=" * 70)
    
    # Create network
    network = MultiRegionNetwork()
    network.add_region(region_id=0, n_synapses=2, name="Source")
    network.add_region(region_id=1, n_synapses=2, name="Target")
    network.connect_regions(0, 1, distance_mm=1.0)
    
    dt = 0.001
    active_stim = {'voltage': -10e-3, 'reward': False}
    
    # Run with detailed tracing
    for step in range(500):
        t_ms = step * dt * 1000
        
        # Manual step with debug output every 100ms
        if step % 100 == 0:
            print(f"\n--- Step {step} (t={t_ms:.0f}ms) ---")
        
        network.time += dt
        
        # Step 1: Run regions
        for region_id, region in network.regions.items():
            region.step(dt, active_stim)
        
        # Step 2: Collect and tag packets
        for region_id, region in network.regions.items():
            packets = region.get_emitted_packets()
            
            if step % 100 == 0:
                print(f"  Region {region_id}: {len(packets)} packets emitted")
            
            for packet in packets:
                synapse_idx = packet.source_synapse_id if hasattr(packet, 'source_synapse_id') else 0
                if synapse_idx < len(region.synapses):
                    synapse = region.synapses[synapse_idx]
                    
                    has_dimer_particles = hasattr(synapse, 'dimer_particles')
                    n_dimers = len(synapse.dimer_particles.dimers) if has_dimer_particles else 0
                    
                    if step % 100 == 0 and packets:
                        print(f"    Synapse {synapse_idx}: has_dimer_particles={has_dimer_particles}, n_dimers={n_dimers}")
                    
                    if has_dimer_particles and synapse.dimer_particles.dimers:
                        packet.dimer_link = network.cross_neuron_entanglement.tag_packet_with_dimers(
                            packet_id=packet.id,
                            emission_time=network.time,
                            source_dimers=synapse.dimer_particles.dimers,
                            source_neuron_id=region_id,
                            source_synapse_id=synapse_idx
                        )
                        packet.source_neuron_id = region_id
                        
                        if step % 100 == 0:
                            n_snapshots = len(packet.dimer_link.dimer_snapshots) if packet.dimer_link else 0
                            print(f"    Tagged packet {packet.id} with {n_snapshots} dimer snapshots")
            
            # Inject into waveguides
            if packets:
                network.waveguide_network.inject_packets(
                    source_id=region_id,
                    packets=packets,
                    current_time=network.time
                )
        
        # Step 3: Propagate
        deliveries = network.waveguide_network.step(network.time)
        
        if step % 100 == 0 and deliveries:
            for tgt, pkts in deliveries.items():
                print(f"  Deliveries to region {tgt}: {len(pkts)} packets")
        
        # Step 4: Receive and process entanglement
        for target_id, packets in deliveries.items():
            if target_id in network.regions:
                region = network.regions[target_id]
                region.receive_photons(packets, network.time)
                
                for packet_record in packets:
                    packet = packet_record.get('packet')
                    has_dimer_link = packet and hasattr(packet, 'dimer_link') and packet.dimer_link is not None
                    
                    if step % 100 == 0:
                        print(f"    Packet has dimer_link: {has_dimer_link}")
                        if has_dimer_link:
                            print(f"      dimer_link snapshots: {len(packet.dimer_link.dimer_snapshots)}")
                    
                    if has_dimer_link:
                        for syn_idx, synapse in enumerate(region.synapses):
                            if hasattr(synapse, 'dimer_particles') and synapse.dimer_particles.dimers:
                                target_dimers = synapse.dimer_particles.dimers
                                
                                if step % 100 == 0:
                                    print(f"      Target synapse {syn_idx} has {len(target_dimers)} dimers")
                                    # Check coincidence window
                                    for td in target_dimers[:3]:  # First 3
                                        age = network.time - td.birth_time
                                        print(f"        Dimer {td.id}: age={age:.3f}s, P_S={td.singlet_probability:.3f}")
                                
                                new_bonds = network.cross_neuron_entanglement.process_arrival(
                                    dimer_link=packet.dimer_link,
                                    n_photons_delivered=packet_record.get('delivered_photons', 0),
                                    target_dimers=target_dimers,
                                    target_neuron_id=target_id,
                                    target_synapse_id=syn_idx,
                                    current_time=network.time
                                )
                                
                                if new_bonds:
                                    print(f"      *** NEW BONDS: {len(new_bonds)} ***")
        
        # Step 5: Decay
        dimer_states = network._collect_dimer_states()
        network.cross_neuron_entanglement.step(dt, dimer_states, network.time)
        
        if step % 100 == 0:
            print(f"  Total bonds: {len(network.cross_neuron_entanglement.bonds)}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Final bonds: {len(network.cross_neuron_entanglement.bonds)}")
    print(f"Total formed: {network.cross_neuron_entanglement.total_bonds_formed}")
    print(f"Total broken: {network.cross_neuron_entanglement.total_bonds_broken}")


if __name__ == "__main__":
    debug_entanglement()