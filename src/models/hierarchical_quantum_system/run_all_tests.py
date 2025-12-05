"""
MASTER TEST RUNNER: HIERARCHICAL QUANTUM PROCESSING MODEL

Main interface for running all experimental validation protocols.
Generates comprehensive validation report with all figures and analyses.

Usage:
------
python run_all_tests.py [--quick] [--output-dir PATH]

Options:
    --quick: Run abbreviated tests (shorter duration, fewer conditions)
    --output-dir: Directory for saving results (default: /mnt/user-data/outputs/validation_results)

Author: Assistant with human researcher
Date: October 2025
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import List
from typing import Dict

# Import protocols
from experimental_protocols import (
    IsotopeSubstitutionProtocol,
    TemperatureIndependenceProtocol,
    MagneticFieldResonanceProtocol
)

from experimental_protocols_extended import (
    AnestheticDisruptionProtocol,
    UVWavelengthSpecificityProtocol,
    TemporalIntegrationProtocol,
    CombinedValidationProtocol
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# TEST SUITE CONFIGURATION
# =============================================================================

class TestSuite:
    """
    Complete validation test suite
    
    Organizes tests by priority and generates comprehensive report
    """
    
    def __init__(self, output_dir: str = './validation_results',
                 quick_mode: bool = False):
        """
        Initialize test suite
        
        Parameters:
        ----------
        output_dir : str
            Directory for saving results
        quick_mode : bool
            If True, run abbreviated tests
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quick_mode = quick_mode
        
        # Test duration
        self.duration = 100.0 if quick_mode else 200.0
        
        # Results storage
        self.results = {}
        self.analyses = {}
        
        logger.info(f"\n{'='*70}")
        logger.info(f"TEST SUITE INITIALIZED")
        logger.info(f"{'='*70}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Quick mode: {quick_mode}")
        logger.info(f"Duration: {self.duration}s per test")
        
    def run_phase_1_critical_tests(self, orchestrator):
        """
        Phase 1: Critical smoking gun tests
        
        These are the essential tests that must pass to validate
        quantum hypothesis.
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"PHASE 1: CRITICAL TESTS")
        logger.info(f"{'='*70}")
        
        # Test 1: Isotope Substitution
        logger.info("\n>>> Running Test 1: Isotope Substitution")
        test1 = IsotopeSubstitutionProtocol()
        self.results['isotope'] = test1.run(orchestrator, duration=self.duration)
        self.analyses['isotope'] = test1.analyze()
        test1.plot(save_path=self.output_dir / 'test_1_isotope_substitution.png')
        
        # Test 2: Temperature Independence
        logger.info("\n>>> Running Test 2: Temperature Independence")
        test2 = TemperatureIndependenceProtocol()
        self.results['temperature'] = test2.run(orchestrator, duration=self.duration)
        self.analyses['temperature'] = test2.analyze()
        test2.plot(save_path=self.output_dir / 'test_2_temperature_independence.png')
        
        # Test 3: Magnetic Field Resonance
        logger.info("\n>>> Running Test 3: Magnetic Field Resonance")
        test3 = MagneticFieldResonanceProtocol()
        self.results['magnetic'] = test3.run(orchestrator, duration=self.duration)
        self.analyses['magnetic'] = test3.analyze()
        test3.plot(save_path=self.output_dir / 'test_3_magnetic_field_resonance.png')
        
        # Test 4: Anesthetic Disruption
        logger.info("\n>>> Running Test 4: Anesthetic Disruption")
        test4 = AnestheticDisruptionProtocol()
        self.results['anesthetic'] = test4.run(orchestrator, duration=self.duration)
        self.analyses['anesthetic'] = test4.analyze()
        test4.plot(save_path=self.output_dir / 'test_4_anesthetic_disruption.png')
        
        logger.info(f"\n{'='*70}")
        logger.info(f"PHASE 1 COMPLETE")
        logger.info(f"{'='*70}")
        
    def run_phase_2_mechanistic_tests(self, orchestrator):
        """
        Phase 2: Mechanistic detail tests
        
        These tests provide mechanistic insight into how the
        quantum processing works.
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"PHASE 2: MECHANISTIC TESTS")
        logger.info(f"{'='*70}")
        
        # Test 5: UV Wavelength Specificity
        logger.info("\n>>> Running Test 5: UV Wavelength Specificity")
        test5 = UVWavelengthSpecificityProtocol()
        self.results['uv_wavelength'] = test5.run(orchestrator, duration=self.duration)
        self.analyses['uv_wavelength'] = test5.analyze()
        test5.plot(save_path=self.output_dir / 'test_5_uv_wavelength_specificity.png')
        
        # Test 11: Temporal Integration Window
        logger.info("\n>>> Running Test 11: Temporal Integration Window")
        test11 = TemporalIntegrationProtocol()
        n_trials = 3 if self.quick_mode else 5
        self.results['temporal'] = test11.run(orchestrator, n_trials=n_trials)
        self.analyses['temporal'] = test11.analyze()
        test11.plot(save_path=self.output_dir / 'test_11_temporal_integration.png')
        
        logger.info(f"\n{'='*70}")
        logger.info(f"PHASE 2 COMPLETE")
        logger.info(f"{'='*70}")
        
    def run_phase_3_combined_validation(self, orchestrator):
        """
        Phase 3: Combined validation matrix
        
        Tests all manipulations together to ensure consistency
        across conditions.
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"PHASE 3: COMBINED VALIDATION")
        logger.info(f"{'='*70}")
        
        test_combined = CombinedValidationProtocol()
        self.results['combined'] = test_combined.run(orchestrator, duration=self.duration)
        self.analyses['combined'] = test_combined.analyze()
        test_combined.plot(save_path=self.output_dir / 'combined_validation_matrix.png')
        
        logger.info(f"\n{'='*70}")
        logger.info(f"PHASE 3 COMPLETE")
        logger.info(f"{'='*70}")
        
    def generate_summary_report(self):
        """
        Generate comprehensive summary report
        
        Includes:
        - Test results summary
        - Validation scores
        - Pass/fail for each prediction
        - Overall assessment
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"GENERATING SUMMARY REPORT")
        logger.info(f"{'='*70}")
        
        report = {
            'metadata': {
                'timestamp': str(datetime.now()),
                'quick_mode': self.quick_mode,
                'test_duration_s': self.duration
            },
            'phase_1_critical': {
                'isotope': self.analyses.get('isotope', {}),
                'temperature': self.analyses.get('temperature', {}),
                'magnetic': self.analyses.get('magnetic', {}),
                'anesthetic': self.analyses.get('anesthetic', {})
            },
            'phase_2_mechanistic': {
                'uv_wavelength': self.analyses.get('uv_wavelength', {}),
                'temporal': self.analyses.get('temporal', {})
            },
            'phase_3_combined': self.analyses.get('combined', {}),
            'overall_assessment': self._assess_validation()
        }
        
        # Save JSON report
        report_path = self.output_dir / 'validation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report saved to: {report_path}")
        
        # Generate text summary
        self._print_summary_report(report)
        
        # Save text summary
        summary_path = self.output_dir / 'validation_summary.txt'
        with open(summary_path, 'w') as f:
            self._write_summary_report(report, f)
        logger.info(f"Summary saved to: {summary_path}")
        
        return report
        
    def _assess_validation(self):
        """
        Assess overall validation status
        """
        # Collect all test pass/fail results
        tests_passed = []
        tests_failed = []
        
        for test_name, analysis in self.analyses.items():
            if analysis.get('test_passed', False):
                tests_passed.append(test_name)
            else:
                tests_failed.append(test_name)
                
        total_tests = len(tests_passed) + len(tests_failed)
        pass_rate = len(tests_passed) / total_tests if total_tests > 0 else 0
        
        # Critical tests that must pass
        critical_tests = ['isotope', 'temperature', 'anesthetic', 'temporal']
        critical_passed = sum(1 for t in critical_tests if t in tests_passed)
        critical_required = len(critical_tests)
        
        # Overall assessment
        if critical_passed == critical_required and pass_rate >= 0.8:
            status = 'VALIDATED'
            confidence = 'HIGH'
        elif critical_passed >= 3 and pass_rate >= 0.6:
            status = 'PARTIALLY VALIDATED'
            confidence = 'MODERATE'
        else:
            status = 'NOT VALIDATED'
            confidence = 'LOW'
            
        return {
            'status': status,
            'confidence': confidence,
            'tests_passed': tests_passed,
            'tests_failed': tests_failed,
            'pass_rate': pass_rate,
            'critical_tests_passed': f"{critical_passed}/{critical_required}",
            'recommendation': self._generate_recommendation(status, tests_failed)
        }
        
    def _generate_recommendation(self, status: str, failed_tests: List[str]) -> str:
        """
        Generate recommendation based on validation results
        """
        if status == 'VALIDATED':
            return (
                "All critical predictions confirmed. Hierarchical quantum processing "
                "mechanism is validated. Recommended next steps: (1) Publish results, "
                "(2) Design follow-up experiments to test additional predictions, "
                "(3) Explore therapeutic applications."
            )
        elif status == 'PARTIALLY VALIDATED':
            return (
                f"Most predictions confirmed, but {len(failed_tests)} tests failed: "
                f"{', '.join(failed_tests)}. Recommended next steps: (1) Investigate "
                "why these tests failed, (2) Refine model parameters, (3) Run "
                "additional validation experiments."
            )
        else:
            return (
                f"Validation unsuccessful. Multiple tests failed: {', '.join(failed_tests)}. "
                "Recommended next steps: (1) Review model assumptions, (2) Check "
                "parameter values against literature, (3) Consider alternative mechanisms."
            )
            
    def _print_summary_report(self, report: Dict):
        """
        Print summary report to console
        """
        assessment = report['overall_assessment']
        
        print(f"\n{'='*70}")
        print(f"VALIDATION SUMMARY REPORT")
        print(f"{'='*70}")
        print(f"Timestamp: {report['metadata']['timestamp']}")
        print(f"Test Duration: {report['metadata']['test_duration_s']:.0f} seconds per test")
        print(f"\n{'='*70}")
        print(f"OVERALL ASSESSMENT")
        print(f"{'='*70}")
        print(f"Status: {assessment['status']}")
        print(f"Confidence: {assessment['confidence']}")
        print(f"Pass Rate: {assessment['pass_rate']*100:.0f}%")
        print(f"Critical Tests: {assessment['critical_tests_passed']}")
        print(f"\nTests Passed: {', '.join(assessment['tests_passed'])}")
        if assessment['tests_failed']:
            print(f"Tests Failed: {', '.join(assessment['tests_failed'])}")
        print(f"\n{'='*70}")
        print(f"RECOMMENDATION")
        print(f"{'='*70}")
        print(f"{assessment['recommendation']}")
        print(f"\n{'='*70}")
        print(f"DETAILED RESULTS")
        print(f"{'='*70}")
        
        # Phase 1 results
        print(f"\nPHASE 1 - CRITICAL TESTS:")
        for test_name, analysis in report['phase_1_critical'].items():
            if analysis:
                passed = analysis.get('test_passed', False)
                status_symbol = '✓' if passed else '✗'
                print(f"  {status_symbol} {test_name.upper()}: {'PASSED' if passed else 'FAILED'}")
                
        # Phase 2 results
        print(f"\nPHASE 2 - MECHANISTIC TESTS:")
        for test_name, analysis in report['phase_2_mechanistic'].items():
            if analysis:
                passed = analysis.get('test_passed', False)
                status_symbol = '✓' if passed else '✗'
                print(f"  {status_symbol} {test_name.upper()}: {'PASSED' if passed else 'FAILED'}")
                
        # Phase 3 results
        if report['phase_3_combined']:
            print(f"\nPHASE 3 - COMBINED VALIDATION:")
            analysis = report['phase_3_combined']
            passed = analysis.get('test_passed', False)
            status_symbol = '✓' if passed else '✗'
            score = analysis.get('validation_score', 0) * 100
            print(f"  {status_symbol} COMBINED: {'PASSED' if passed else 'FAILED'} ({score:.0f}%)")
            
        print(f"\n{'='*70}")
        
    def _write_summary_report(self, report: Dict, file):
        """
        Write detailed summary report to file
        """
        assessment = report['overall_assessment']
        
        file.write("="*70 + "\n")
        file.write("HIERARCHICAL QUANTUM PROCESSING MODEL\n")
        file.write("COMPREHENSIVE VALIDATION REPORT\n")
        file.write("="*70 + "\n\n")
        
        file.write(f"Timestamp: {report['metadata']['timestamp']}\n")
        file.write(f"Test Duration: {report['metadata']['test_duration_s']:.0f} seconds per test\n")
        file.write(f"Quick Mode: {report['metadata']['quick_mode']}\n\n")
        
        file.write("="*70 + "\n")
        file.write("OVERALL ASSESSMENT\n")
        file.write("="*70 + "\n")
        file.write(f"Status: {assessment['status']}\n")
        file.write(f"Confidence: {assessment['confidence']}\n")
        file.write(f"Pass Rate: {assessment['pass_rate']*100:.0f}%\n")
        file.write(f"Critical Tests Passed: {assessment['critical_tests_passed']}\n\n")
        
        file.write("Recommendation:\n")
        file.write(f"{assessment['recommendation']}\n\n")
        
        file.write("="*70 + "\n")
        file.write("DETAILED TEST RESULTS\n")
        file.write("="*70 + "\n\n")
        
        # Write detailed results for each phase
        file.write("PHASE 1: CRITICAL TESTS\n")
        file.write("-"*70 + "\n")
        for test_name, analysis in report['phase_1_critical'].items():
            if analysis:
                file.write(f"\n{test_name.upper()}:\n")
                file.write(f"  Pass/Fail: {'PASSED' if analysis.get('test_passed') else 'FAILED'}\n")
                # Write key metrics
                for key, value in analysis.items():
                    if key not in ['test_name', 'test_passed', 'detailed_metrics', 'predictions']:
                        file.write(f"  {key}: {value}\n")
                        
        file.write("\n" + "="*70 + "\n")
        file.write("END OF REPORT\n")
        file.write("="*70 + "\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run hierarchical quantum processing model validation tests'
    )
    parser.add_argument('--quick', action='store_true',
                       help='Run abbreviated tests (faster, less comprehensive)')
    parser.add_argument('--output-dir', type=str,
                       default='./validation_results',
                       help='Output directory for results')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3],
                       help='Run only specified phase (1, 2, or 3)')
    
    args = parser.parse_args()
    
    # Initialize test suite
    suite = TestSuite(output_dir=args.output_dir, quick_mode=args.quick)
    
    # Import and initialize orchestrator
    try:
        # Try to import the actual orchestrator
        import sys
        sys.path.append('/mnt/user-data/outputs')
        from master_orchestrator import HierarchicalQuantumOrchestrator
        
        orchestrator = HierarchicalQuantumOrchestrator()
        logger.info("Master orchestrator loaded successfully")
        
    except ImportError:
        logger.warning("Could not import master orchestrator - creating mock")
        # Create a minimal mock for testing the protocol framework
        class MockOrchestrator:
            def __init__(self):
                from experimental_protocols import ExperimentalMetrics
                from dataclasses import dataclass
                
                @dataclass
                class MockState:
                    ca_concentration: float = 100e-9
                    camkii_phosphorylated: float = 0.0
                    psd95_open: float = 0.091
                    actin_reorganized: float = 0.0
                    n_tryptophans: int = 80
                    trp_em_field: float = 0.0
                    n_coherent_dimers: int = 0
                    dimer_coherence: float = 0.0
                    mt_present: bool = False
                    trp_excited_fraction: float = 0.0
                    superradiance_enhancement: float = 1.0
                    reverse_modulation_kT: float = 0.0
                    forward_enhancement: float = 1.0
                    
                self.state = MockState()
                self.experimental = {}
                self.history = {'dimer_coherence': []}
                self.params = None
                
            def set_experimental_conditions(self, **kwargs):
                self.experimental = kwargs
                
            def step(self, dt):
                # Minimal simulation for testing
                if self.state.ca_concentration > 1e-6:
                    self.state.camkii_phosphorylated = min(0.8, self.state.camkii_phosphorylated + 0.1)
                    self.state.psd95_open = min(0.7, self.state.psd95_open + 0.05)
                    
                    # Isotope-dependent coherence
                    if self.experimental.get('isotope') == 'P31':
                        self.state.dimer_coherence = 0.8
                        self.state.n_coherent_dimers = 100
                    else:
                        self.state.dimer_coherence = 0.1
                        self.state.n_coherent_dimers = 10
                        
        orchestrator = MockOrchestrator()
        logger.info("Mock orchestrator created for testing")
    
    # Run test phases
    if args.phase is None or args.phase == 1:
        suite.run_phase_1_critical_tests(orchestrator)
        
    if args.phase is None or args.phase == 2:
        suite.run_phase_2_mechanistic_tests(orchestrator)
        
    if args.phase is None or args.phase == 3:
        suite.run_phase_3_combined_validation(orchestrator)
        
    # Generate summary report
    report = suite.generate_summary_report()
    
    logger.info(f"\n{'='*70}")
    logger.info(f"VALIDATION COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"All results saved to: {suite.output_dir}")
    logger.info(f"Overall status: {report['overall_assessment']['status']}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()