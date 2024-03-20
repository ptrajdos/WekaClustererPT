package weka.clusterers;

import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
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
		int numClasses = data.numClasses();

		
		Clusterer clust = this.getClusterer();
		try {
			clust.buildClusterer(data);
			int numberOfClusters = clust.numberOfClusters();
			int desiredNumberOfClusters = numClasses * 2;
			ClassSpecificClusterer cClust = (ClassSpecificClusterer)clust;
			assertTrue("Wrong number of clusters for the default implementation", numberOfClusters == desiredNumberOfClusters);
			int[][] instancesCounts = new int[numClasses][2];
			
			for (Instance instance : data) {
				double[] distribution = clust.distributionForInstance(instance);
				assertTrue("Distribution length", numberOfClusters == distribution.length);
				assertTrue("Distribution check", DistributionChecker.checkDistribution(distribution));
				
				
				double[][] classSpecDistribution = cClust.classSpecificDistributionForInstance(instance);
				
				int classIdx =(int) instance.classValue();
				int selClusterIdx = classSpecDistribution[classIdx][0] > classSpecDistribution[classIdx][1]? 0:1;
				instancesCounts[classIdx][selClusterIdx]+=1;
				
				assertTrue("Number of classes", cClust.numberOfClasses() == 2);
				assertTrue("Number of classes for class-specific distribution", classSpecDistribution.length == 2);
				assertTrue("Number of clusters for class-specific distribution", classSpecDistribution[0].length == 2);
				assertTrue("Number of clusters for class-specific distribution", classSpecDistribution[1].length == 2);
				assertTrue("Class specific distribution check", DistributionChecker.checkDistribution(classSpecDistribution[0]));
				assertTrue("Class specific distribution check", DistributionChecker.checkDistribution(classSpecDistribution[1]));
				
			}
			for(int c=0; c<numClasses;c++) {
				for(int j=0;j<2;j++) {
					assertTrue("Empty final cluster: ["+ c +";" + j + "]",instancesCounts[c][j]>0);
				}
			}

			double[][][] classSpecDistributionAll = cClust.classSpecificDistrbutionForInstances(data);
			assertTrue("Number of predicted class-specific instances", classSpecDistributionAll.length == data.numInstances());


		} catch (Exception e) {
			fail("An exception has been caught: " + e.getMessage());
		} 
	}
	
	public void testClassDataSmallSample() {
		RandomDataGenerator gen = new RandomDataGenerator();
		gen.setNumObjects(4);
		Instances data = gen.generateData();
		int numClasses = data.numClasses();

		
		Clusterer clust = this.getClusterer();
		try {
			clust.buildClusterer(data);
			int numberOfClusters = clust.numberOfClusters();
			int desiredNumberOfClusters = numClasses * 2;
			ClassSpecificClusterer cClust = (ClassSpecificClusterer)clust;
			assertTrue("Wrong number of clusters for the default implementation", numberOfClusters == desiredNumberOfClusters);
			int[][] instancesCounts = new int[numClasses][2];
			
			for (Instance instance : data) {
				double[] distribution = clust.distributionForInstance(instance);
				assertTrue("Distribution length", numberOfClusters == distribution.length);
				assertTrue("Distribution check", DistributionChecker.checkDistribution(distribution));
				
				
				double[][] classSpecDistribution = cClust.classSpecificDistributionForInstance(instance);
				
				int classIdx =(int) instance.classValue();
				int selClusterIdx = classSpecDistribution[classIdx][0] > classSpecDistribution[classIdx][1]? 0:1;
				instancesCounts[classIdx][selClusterIdx]+=1;
				
				assertTrue("Number of classes", cClust.numberOfClasses() == 2);
				assertTrue("Number of classes for class-specific distribution", classSpecDistribution.length == 2);
				assertTrue("Number of clusters for class-specific distribution", classSpecDistribution[0].length == 2);
				assertTrue("Number of clusters for class-specific distribution", classSpecDistribution[1].length == 2);
				assertTrue("Class specific distribution check", DistributionChecker.checkDistribution(classSpecDistribution[0]));
				assertTrue("Class specific distribution check", DistributionChecker.checkDistribution(classSpecDistribution[1]));
				
			}
			for(int c=0; c<numClasses;c++) {
				for(int j=0;j<2;j++) {
					assertTrue("Empty final cluster: ["+ c +";" + j + "]",instancesCounts[c][j]>0);
				}
			}

			double[][][] classSpecDistributionAll = cClust.classSpecificDistrbutionForInstances(data);
			assertTrue("Number of predicted class-specific instances", classSpecDistributionAll.length == data.numInstances());


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
		ClassSpecificClusterer cClust = (ClassSpecificClusterer)clust;
		try {
			clust.buildClusterer(data);
			int numberOfClusters = clust.numberOfClusters();
			int nClasses = data.numClasses();
			for (Instance instance : testData) {
				double[] distribution = clust.distributionForInstance(instance);
				assertTrue("Distribution length", numberOfClusters == distribution.length);
				assertTrue("Distribution check", DistributionChecker.checkDistribution(distribution));
				double[][] classSpecDistribution = cClust.classSpecificDistributionForInstance(instance);
				assertTrue("Number of classes", cClust.numberOfClasses() == nClasses);
				assertTrue("No instances class-specific cluster response length", classSpecDistribution.length == nClasses);
				assertTrue("No instances class-specific cluster response", classSpecDistribution[0].length == 1);
				for( int c =0; c<nClasses; c++)
					assertTrue("No instances class-specific cluster response -- value", Utils.eq(classSpecDistribution[c][0], 1) );

			}
		} catch (Exception e) {
			fail("An exception has been caught: " + e.getMessage());
		} 
	}

	public void testNoClassData(){
		Clusterer clusterer = this.getClusterer();
		ClassSpecificClusterer cClust = (ClassSpecificClusterer)clusterer;

		RandomDataGenerator gen = new RandomDataGenerator();
		gen.setNumNominalAttributes(0);
		gen.setNumStringAttributes(0);
		gen.setNumDateAttributes(0);
		gen.setAddClassAttrib(false);

		Instances data = gen.generateData();

		try{
			clusterer.buildClusterer(data);
			for (Instance instance : data) {
				double[] distribution = clusterer.distributionForInstance(instance);
				int numberOfClusters = clusterer.numberOfClusters();

				assertTrue("Distribution length", numberOfClusters == distribution.length);
				assertTrue("Distribution check", DistributionChecker.checkDistribution(distribution));

				double[][] classSpecDistribution = cClust.classSpecificDistributionForInstance(instance);
				assertTrue("Number of classes", cClust.numberOfClasses() == 0);
				assertTrue("Number of classes for class-specific distribution", classSpecDistribution.length == 1);
				assertTrue("Number of clusters for class-specific distribution", classSpecDistribution[0].length == 2);
				assertTrue("Class specific distribution check", DistributionChecker.checkDistribution(classSpecDistribution[0]));
			}

		}catch(Exception e){
			fail("An exception has been caught: " + e);
		}
	}

	public void testOnClassOnly(){
		Clusterer clusterer = this.getClusterer();
		ClassSpecificClusterer cClust = (ClassSpecificClusterer)clusterer;

		RandomDataGenerator gen = new RandomDataGenerator();
		gen.setNumNominalAttributes(0);
		gen.setNumStringAttributes(0);
		gen.setNumDateAttributes(0);
		gen.setAddClassAttrib(true);
		gen.setNumNumericAttributes(0);

		Instances data = gen.generateData();
		int nClasses = data.numClasses();

		try{
			clusterer.buildClusterer(data);

			int[] classSpecCluserNum = cClust.numberOfClassSpecificClusters();
			assertTrue("Number of class-specific clusters", classSpecCluserNum.length == nClasses);
			for(int c = 0; c<classSpecCluserNum.length;c++){
				assertTrue("Per class cluster number", classSpecCluserNum[c] == 1);
			}
			for (Instance instance : data) {
				double[] distribution = clusterer.distributionForInstance(instance);
				int numberOfClusters = clusterer.numberOfClusters();

				assertTrue("Distribution length", numberOfClusters == distribution.length);
				assertTrue("Distribution check", DistributionChecker.checkDistribution(distribution));

				double[][] classSpecDistribution = cClust.classSpecificDistributionForInstance(instance);
				assertTrue("Number of classes", cClust.numberOfClasses() == data.numClasses());
				int numClasses = data.numClasses();
				assertTrue("Number of classes for class-specific distribution", classSpecDistribution.length == numClasses );
				for(int c=0; c<numClasses;c++){
					assertTrue("Number of clusters for class-specific distribution", classSpecDistribution[c].length == 1);
					assertTrue("Class specific distribution check", DistributionChecker.checkDistribution(classSpecDistribution[c]));
				}
				
			}

		}catch(Exception e){
			fail("An exception has been caught: " + e);
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
