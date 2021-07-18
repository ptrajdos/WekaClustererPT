package weka.clusterers;

import java.util.List;
import java.util.Random;

import org.apache.commons.math3.util.Pair;

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
	
	public void testDistanceList() {
		 XmeansWithKmeansPP clusterer = (XmeansWithKmeansPP) this.getClusterer();
		 RandomDataGenerator gen = new RandomDataGenerator();
		 gen.setNumNominalAttributes(0);
		 gen.setAddClassAttrib(false);
		 
		 Instances data = gen.generateData();
		 Instance tmpInstance = data.get(0);
		 
		 try {
			clusterer.buildClusterer(data);
		} catch (Exception e) {
			fail("An exception has been thrown!" + e.getMessage());
		}
		 
		 List<Pair<Instance,Double>> sqDistList =clusterer.getSquaredDistances(tmpInstance);
		 
		 assertTrue("List not null", sqDistList !=null);
		 
		 int listLen = sqDistList.size();
		 boolean increasingOrder=true;
		 
		 for(int i=0 ;i<listLen-1;i++) {
			 if(sqDistList.get(i).getValue() < sqDistList.get(i+1).getValue()) {
				 increasingOrder=false;
				 break;
			 }	 
		 }
		 
		 assertTrue("Decreasing order", increasingOrder);
		 
	}
	
	public void testRouleteSelector() {
		XmeansWithKmeansPP clusterer = (XmeansWithKmeansPP) this.getClusterer();
		 RandomDataGenerator gen = new RandomDataGenerator();
		 gen.setNumNominalAttributes(0);
		 gen.setAddClassAttrib(false);
		 
		 Instances data = gen.generateData();
		 Instance tmpInstance = data.get(0);
		 
		 Instance selectedInstance = null;
		 Random rnd = new Random(0);
		 try {
			clusterer.buildClusterer(data);
			List<Pair<Instance,Double>> sqDistList =clusterer.getSquaredDistances(tmpInstance);
			selectedInstance = clusterer.rouletteSelector(sqDistList, rnd);
			
			assertTrue("Selected not null", selectedInstance!=null);
			
		} catch (Exception e) {
			fail("An exception has been thrown!" + e.getMessage());
		}
	}
	
	public void testDateAttribs() {
		XmeansWithKmeansPP clusterer = (XmeansWithKmeansPP) this.getClusterer();
		 RandomDataGenerator gen = new RandomDataGenerator();
		 gen.setNumNominalAttributes(0);
		 gen.setNumNumericAttributes(0);
		 gen.setNumDateAttributes(4);
		 gen.setAddClassAttrib(false);
		 
		 Instances data = gen.generateData();
		 
		 try {
			clusterer.buildClusterer(data);
		} catch (Exception e) {
			fail("An exception has been caught:" + e.getMessage());
		}
	}


}
