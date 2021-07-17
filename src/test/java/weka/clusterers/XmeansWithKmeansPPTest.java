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

public class XmeansWithKmeansPPTest extends AbstractClustererTest {

	public XmeansWithKmeansPPTest(String name) {
		super(name);
		// TODO Auto-generated constructor stub
	}


	@Override
	public Clusterer getClusterer() {
		return new XmeansWithKmeansPP();
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
			clusterer.buildClusterer(dataset);;
			for (Instance instance : dataset) {
				double[] distribution = clusterer.distributionForInstance(instance);
				assertTrue("Check distribution", DistributionChecker.checkDistribution(distribution));
			}
			
		} catch (Exception e) {
			fail("An exception has been caught " + e.getMessage());
		}
	 }


}
