package weka.clusterers;

import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.tools.data.RandomDataGenerator;
import weka.tools.data.RandomDoubleGenerator;
import weka.tools.data.RandomDoubleGeneratorGaussian;
import weka.tools.tests.DistributionChecker;
import weka.tools.tests.WekaGOEChecker;

public class ClassSpecificClustererTest extends AbstractClustererTest {

	public ClassSpecificClustererTest(String name) {
		super(name);
	}

	@Override
	public Clusterer getClusterer() {
		return new ClassSpecificClusterer();
	}
	
	public void testTipTexts() {
		WekaGOEChecker check = new WekaGOEChecker();
		check.setObject(this.getClusterer());
		if(check.checkGlobalInfo())
			assertTrue("Global Info call", check.checkCallGlobalInfo());
		
		if(check.checkToolTips())
			assertTrue("Tip Texts call", check.checkToolTipsCall());
	}
	
	public void testClassData() {
		RandomDataGenerator gen = new RandomDataGenerator();
		Instances data = gen.generateData();
		
		Clusterer clust = this.getClusterer();
		try {
			clust.buildClusterer(data);
			int numberOfClusters = clust.numberOfClusters();
			for (Instance instance : data) {
				double[] distribution = clust.distributionForInstance(instance);
				assertTrue("Distribution length", numberOfClusters == distribution.length);
				assertTrue("Distribution check", DistributionChecker.checkDistribution(distribution));
			}
		} catch (Exception e) {
			fail("An exception has been caught: " + e.getMessage());
		} 
	}
	
	public void testNoInstances(){
		RandomDataGenerator gen = new RandomDataGenerator();
		gen.setNumObjects(0);
		Instances data = gen.generateData();
		
		gen.setNumObjects(10);
		Instances testData = gen.generateData();
		
		Clusterer clust = this.getClusterer();
		try {
			clust.buildClusterer(data);
			int numberOfClusters = clust.numberOfClusters();
			for (Instance instance : testData) {
				double[] distribution = clust.distributionForInstance(instance);
				assertTrue("Distribution length", numberOfClusters == distribution.length);
				assertTrue("Distribution check", DistributionChecker.checkDistribution(distribution));
			}
		} catch (Exception e) {
			fail("An exception has been caught: " + e.getMessage());
		} 
	}
	
	public void testOnCondensedData() {
		 Clusterer clusterer = this.getClusterer();
		 RandomDataGenerator gen = new RandomDataGenerator();
		 gen.setNumNominalAttributes(0);
		 gen.setNumStringAttributes(0);
		 gen.setNumDateAttributes(0);
		 RandomDoubleGenerator doubleGen = new RandomDoubleGeneratorGaussian();
		 doubleGen.setDivisor(10000.0);
		 gen.setDoubleGen(doubleGen );
		 
		 Instances dataset = gen.generateData();
		 try {
			clusterer.buildClusterer(dataset);;
			for (Instance instance : dataset) {
				double[] distribution = clusterer.distributionForInstance(instance);
				assertTrue("Check distribution", DistributionChecker.checkDistribution(distribution));
			}
			
		} catch (Exception e) {
			fail("An exception has been caught " + e.getMessage());
		}
	 }

	
	public void testCapabilities() {
		Clusterer clust = this.getClusterer();
		Capabilities caps  = clust.getCapabilities();
		assertTrue("No class capaiblity disabled", !caps.handles(Capability.NO_CLASS));
		RandomDataGenerator gen = new RandomDataGenerator();
		Instances data = gen.generateData();
		
		try {
			caps.testWithFail(data);
		} catch (Exception e) {
			fail("Capabilities checking against the data has failed: " + e.getMessage());
		}
		
	}



}
