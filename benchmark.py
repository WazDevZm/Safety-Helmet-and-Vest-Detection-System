"""
Benchmark script for Safety Helmet and Vest Detection System
Performance testing and optimization analysis
"""

import time
import cv2
import numpy as np
from ppe_detector import PPEDetector
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import psutil
import os

class PerformanceBenchmark:
    """Performance benchmarking for PPE detection system"""
    
    def __init__(self):
        self.detector = PPEDetector()
        self.results = []
    
    def create_test_images(self, num_images: int = 10) -> List[np.ndarray]:
        """Create test images of varying complexity"""
        test_images = []
        
        for i in range(num_images):
            # Create images with different numbers of workers
            num_workers = (i % 3) + 1  # 1, 2, or 3 workers
            
            # Create base image
            img = np.ones((400, 600 * num_workers, 3), dtype=np.uint8) * 255
            
            for j in range(num_workers):
                x_offset = j * 200
                
                # Randomly determine if worker has PPE
                has_helmet = np.random.random() > 0.3
                has_vest = np.random.random() > 0.3
                
                # Head with optional helmet
                if has_helmet:
                    cv2.circle(img, (100 + x_offset, 100), 30, (255, 255, 0), -1)  # Yellow helmet
                cv2.circle(img, (100 + x_offset, 100), 25, (0, 0, 0), -1)  # Black head
                
                # Body with optional vest
                if has_vest:
                    cv2.rectangle(img, (70 + x_offset, 130), (130 + x_offset, 250), (255, 255, 0), -1)  # Yellow vest
                cv2.rectangle(img, (80 + x_offset, 140), (120 + x_offset, 240), (0, 0, 0), -1)  # Black body
                
                # Arms and legs
                cv2.rectangle(img, (50 + x_offset, 140), (70 + x_offset, 200), (0, 0, 0), -1)
                cv2.rectangle(img, (130 + x_offset, 140), (150 + x_offset, 200), (0, 0, 0), -1)
                cv2.rectangle(img, (80 + x_offset, 250), (100 + x_offset, 350), (0, 0, 0), -1)
                cv2.rectangle(img, (100 + x_offset, 250), (120 + x_offset, 350), (0, 0, 0), -1)
            
            test_images.append(img)
        
        return test_images
    
    def benchmark_detection(self, images: List[np.ndarray]) -> Dict[str, Any]:
        """Run detection benchmark on test images"""
        print("🔍 Running PPE detection benchmark...")
        
        processing_times = []
        compliance_rates = []
        detection_counts = []
        
        for i, image in enumerate(images):
            print(f"  Processing image {i+1}/{len(images)}...")
            
            # Measure processing time
            start_time = time.time()
            results = self.detector.detect_ppe(image)
            processing_time = time.time() - start_time
            
            # Record metrics
            processing_times.append(processing_time)
            compliance_rates.append(results['ppe_compliance']['compliance_rate'])
            detection_counts.append(results['total_persons'])
            
            # System resources
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            self.results.append({
                'image_id': i,
                'processing_time': processing_time,
                'compliance_rate': results['ppe_compliance']['compliance_rate'],
                'total_persons': results['total_persons'],
                'compliant_persons': results['ppe_compliance']['compliant_persons'],
                'cpu_usage': cpu_percent,
                'memory_usage': memory_percent
            })
        
        return {
            'processing_times': processing_times,
            'compliance_rates': compliance_rates,
            'detection_counts': detection_counts,
            'avg_processing_time': np.mean(processing_times),
            'std_processing_time': np.std(processing_times),
            'min_processing_time': np.min(processing_times),
            'max_processing_time': np.max(processing_times)
        }
    
    def generate_performance_report(self, benchmark_results: Dict[str, Any]):
        """Generate comprehensive performance report"""
        print("\n📊 Performance Benchmark Results")
        print("=" * 50)
        
        # Basic statistics
        print(f"⏱️  Average Processing Time: {benchmark_results['avg_processing_time']:.3f}s")
        print(f"📈 Standard Deviation: {benchmark_results['std_processing_time']:.3f}s")
        print(f"⚡ Fastest Processing: {benchmark_results['min_processing_time']:.3f}s")
        print(f"🐌 Slowest Processing: {benchmark_results['max_processing_time']:.3f}s")
        
        # Performance rating
        avg_time = benchmark_results['avg_processing_time']
        if avg_time < 1.0:
            rating = "🚀 Excellent"
        elif avg_time < 2.0:
            rating = "✅ Good"
        elif avg_time < 5.0:
            rating = "⚠️ Acceptable"
        else:
            rating = "❌ Needs Optimization"
        
        print(f"🏆 Performance Rating: {rating}")
        
        # System resources
        avg_cpu = np.mean([r['cpu_usage'] for r in self.results])
        avg_memory = np.mean([r['memory_usage'] for r in self.results])
        
        print(f"💻 Average CPU Usage: {avg_cpu:.1f}%")
        print(f"🧠 Average Memory Usage: {avg_memory:.1f}%")
        
        # Compliance analysis
        avg_compliance = np.mean(benchmark_results['compliance_rates'])
        print(f"🦺 Average Compliance Rate: {avg_compliance:.1f}%")
        
        # Detection efficiency
        total_detections = sum(benchmark_results['detection_counts'])
        total_time = sum(benchmark_results['processing_times'])
        detections_per_second = total_detections / total_time if total_time > 0 else 0
        
        print(f"👥 Detections per Second: {detections_per_second:.2f}")
    
    def create_performance_plots(self, benchmark_results: Dict[str, Any]):
        """Create performance visualization plots"""
        print("\n📈 Generating performance plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Processing time distribution
        axes[0, 0].hist(benchmark_results['processing_times'], bins=10, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Processing Time Distribution')
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(benchmark_results['avg_processing_time'], color='red', linestyle='--', 
                          label=f'Average: {benchmark_results["avg_processing_time"]:.3f}s')
        axes[0, 0].legend()
        
        # Compliance rate distribution
        axes[0, 1].hist(benchmark_results['compliance_rates'], bins=10, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Compliance Rate Distribution')
        axes[0, 1].set_xlabel('Compliance Rate (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(benchmark_results['compliance_rates']), color='red', linestyle='--',
                          label=f'Average: {np.mean(benchmark_results["compliance_rates"]):.1f}%')
        axes[0, 1].legend()
        
        # Processing time vs number of workers
        axes[1, 0].scatter(benchmark_results['detection_counts'], benchmark_results['processing_times'], 
                          alpha=0.7, color='orange')
        axes[1, 0].set_title('Processing Time vs Number of Workers')
        axes[1, 0].set_xlabel('Number of Workers')
        axes[1, 0].set_ylabel('Processing Time (seconds)')
        
        # System resource usage
        cpu_usage = [r['cpu_usage'] for r in self.results]
        memory_usage = [r['memory_usage'] for r in self.results]
        
        axes[1, 1].plot(cpu_usage, label='CPU Usage', color='blue')
        axes[1, 1].plot(memory_usage, label='Memory Usage', color='red')
        axes[1, 1].set_title('System Resource Usage')
        axes[1, 1].set_xlabel('Image Index')
        axes[1, 1].set_ylabel('Usage (%)')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def run_benchmark(self, num_images: int = 10):
        """Run complete benchmark suite"""
        print("🚀 Starting PPE Detection Performance Benchmark")
        print("=" * 60)
        
        # Create test images
        print(f"📸 Creating {num_images} test images...")
        test_images = self.create_test_images(num_images)
        
        # Run benchmark
        benchmark_results = self.benchmark_detection(test_images)
        
        # Generate report
        self.generate_performance_report(benchmark_results)
        
        # Create plots
        self.create_performance_plots(benchmark_results)
        
        print("\n✅ Benchmark completed successfully!")
        
        return benchmark_results

def main():
    """Main benchmark function"""
    benchmark = PerformanceBenchmark()
    results = benchmark.run_benchmark(num_images=15)
    
    print("\n🎯 Benchmark Summary:")
    print(f"  📊 Total Images Processed: {len(benchmark.results)}")
    print(f"  ⏱️  Average Processing Time: {results['avg_processing_time']:.3f}s")
    print(f"  🦺 Average Compliance Rate: {np.mean(results['compliance_rates']):.1f}%")
    print(f"  👥 Total Workers Detected: {sum(results['detection_counts'])}")

if __name__ == "__main__":
    main()
