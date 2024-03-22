package weka.clusterers;

import java.util.List;
import java.util.Random;

import org.apache.commons.math3.util.Pair;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.tools.data.RandomDataGenerator;
import weka.tools.data.RandomDoubleGenerator;
import weka.tools.data.RandomDoubleGeneratorGaussian;
import weka.tools.tests.DistributionChecker;
import weka.tools.tests.WekaGOEChecker;

public class XMeansFixedTest extends AbstractClustererTest {

	public XMeansFixedTest(String name) {
		super(name);
		// TODO Auto-generated constructor stub
	}


	@Override
	public Clusterer getClusterer() {
		return new XMeansFixed();
	}
	
	public void testTipTexts() {
		WekaGOEChecker check = new WekaGOEChecker();
		check.setObject(this.getClusterer());
		if(check.checkGlobalInfo())
			assertTrue("Global Info call", check.checkCallGlobalInfo());
		
		if(check.checkToolTips())
			assertTrue("Tip Texts call", check.checkToolTipsCall());
	}
	
	
	
	
	public void testOnCondensedData() {
		 Clusterer clusterer = this.getClusterer();
		 RandomDataGenerator gen = new RandomDataGenerator();
		 gen.setNumNominalAttributes(0);
		 gen.setNumStringAttributes(0);
		 gen.setNumDateAttributes(0);
		 gen.setAddClassAttrib(false);
		 RandomDoubleGenerator doubleGen = new RandomDoubleGeneratorGaussian();
		 doubleGen.setDivisor(10000.0);
		 gen.setDoubleGen(doubleGen );
		 
		 Instances dataset = gen.generateData();
		 try {
			clusterer.buildClusterer(dataset);
			for (Instance instance : dataset) {
				double[] distribution = clusterer.distributionForInstance(instance);
				assertTrue("Check distribution", DistributionChecker.checkDistribution(distribution));
			}
			
		} catch (Exception e) {
			fail("An exception has been caught " + e.getMessage());
		}
	 }
	
	public void testReclusterSamplesSmallSample() {
		 Clusterer clusterer = this.getClusterer();
		 RandomDataGenerator gen = new RandomDataGenerator();
		 gen.setNumNominalAttributes(0);
		 gen.setNumStringAttributes(0);
		 gen.setNumDateAttributes(0);
		 gen.setAddClassAttrib(false);
		 gen.setNumObjects(4);
		 
		 
		 Instances dataset = gen.generateData();
		 try {
			clusterer.buildClusterer(dataset);
			int nClusters = clusterer.numberOfClusters();
			int[] clusterCounts = new int [nClusters];
			
			for (Instance instance : dataset) {
				double[] distribution = clusterer.distributionForInstance(instance);
				int winnerIdx = Utils.maxIndex(distribution);
				clusterCounts[winnerIdx]++;
				assertTrue("Check distribution", DistributionChecker.checkDistribution(distribution));
			}
			
			for(int i =0 ;i<clusterCounts.length;i++) {
				assertTrue("Empty cluster!", clusterCounts[i]>0);
			}
			
		} catch (Exception e) {
			fail("An exception has been caught " + e.getMessage());
		}
	 }
	
	public void testReclusterSamplesOneSample() {
		 Clusterer clusterer = this.getClusterer();
		 RandomDataGenerator gen = new RandomDataGenerator();
		 gen.setNumNominalAttributes(0);
		 gen.setNumStringAttributes(0);
		 gen.setNumDateAttributes(0);
		 gen.setAddClassAttrib(false);
		 gen.setNumObjects(1);
		 
		 
		 Instances trainingDataset = gen.generateData();
		 gen.setNumObjects(100);
		 Instances testingDataset = gen.generateData();
		 try {
			clusterer.buildClusterer(trainingDataset);
			int nClusters = clusterer.numberOfClusters();
			int[] clusterCounts = new int [nClusters];
			
			for (Instance instance : testingDataset) {
				double[] distribution = clusterer.distributionForInstance(instance);//Skewed distr [1;0]
				int winnerIdx = Utils.maxIndex(distribution);
				clusterCounts[winnerIdx]++;
				assertTrue("Check distribution", DistributionChecker.checkDistribution(distribution));
			}
			
			for(int i =0 ;i<clusterCounts.length;i++) {
				assertTrue("Empty cluster!", clusterCounts[i]>0);
			}
			
		} catch (Exception e) {
			fail("An exception has been caught " + e.getMessage());
		}
	 }
	
	public void testReclusterSamplesOneSampleDuplicated() {
		 Clusterer clusterer = this.getClusterer();
		 RandomDataGenerator gen = new RandomDataGenerator();
		 gen.setNumNominalAttributes(0);
		 gen.setNumStringAttributes(0);
		 gen.setNumDateAttributes(0);
		 gen.setAddClassAttrib(false);
		 gen.setNumObjects(1);
		 
		 
		 Instances trainingDataset = gen.generateData();
		 trainingDataset.add(trainingDataset.get(0));//Two duplicated instances
		 
		 gen.setNumObjects(100);
		 Instances testingDataset = gen.generateData();
		 try {
			clusterer.buildClusterer(trainingDataset);
			int nClusters = clusterer.numberOfClusters();
			int[] clusterCounts = new int [nClusters];
			
			for (Instance instance : testingDataset) {
				double[] distribution = clusterer.distributionForInstance(instance);
				int winnerIdx = Utils.maxIndex(distribution);
				clusterCounts[winnerIdx]++;
				assertTrue("Check distribution", DistributionChecker.checkDistribution(distribution));
			}
			
			for(int i =0 ;i<clusterCounts.length;i++) {
				assertTrue("Empty cluster!", clusterCounts[i]>0);
			}
			
		} catch (Exception e) {
			fail("An exception has been caught " + e.getMessage());
		}
	 }
	
	public void testReclusterSamples() {
		 Clusterer clusterer = this.getClusterer();
		 RandomDataGenerator gen = new RandomDataGenerator();
		 gen.setNumNominalAttributes(0);
		 gen.setNumStringAttributes(0);
		 gen.setNumDateAttributes(0);
		 gen.setAddClassAttrib(false);
		 gen.setNumObjects(100);
		 
		 
		 Instances dataset = gen.generateData();
		 try {
			clusterer.buildClusterer(dataset);
			int nClusters = clusterer.numberOfClusters();
			int[] clusterCounts = new int [nClusters];
			
			for (Instance instance : dataset) {
				double[] distribution = clusterer.distributionForInstance(instance);
				int winnerIdx = Utils.maxIndex(distribution);
				clusterCounts[winnerIdx]++;
				assertTrue("Check distribution", DistributionChecker.checkDistribution(distribution));
			}
			
			for(int i =0 ;i<clusterCounts.length;i++) {
				assertTrue("Empty cluster!", clusterCounts[i]>0);
			}
			
		} catch (Exception e) {
			fail("An exception has been caught " + e.getMessage());
		}
	 }
	

}
