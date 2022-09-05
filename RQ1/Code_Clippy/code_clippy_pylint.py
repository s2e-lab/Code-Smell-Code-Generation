import os
import json
from pylint import epylint as elint
import shutil


def analyze(basedir, filename):
    if basedir[-1] != '/':
        basedir += '/'
    outputDir = basedir+'../'+fileName+'_pylint_data'
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
        print("directory \'" + outputDir + "\' created.")
    totalCount = 0
    result = []
    for (root, dirs, file) in os.walk(basedir):
        for f in file:
            # print(f)
            try:
                pylint_stdout, pylint_stderr = elint.py_run(
                    basedir+f + ' --output-format=json', return_std=True)
                # print(pylint_stdout.getvalue())

                result.append(json.loads(pylint_stdout.getvalue()))
            except:
                print('error in file: ' + f)
            totalCount += 1
            if totalCount % 100 == 0:
                print(totalCount)
                with open(outputDir+'/'+str(totalCount)+'.json', 'w') as f:
                    f.write(json.dumps(result))
                result = []

    if len(result) > 0:
        with open(outputDir+'/'+str(totalCount)+'.json', 'w') as f:
            f.write(json.dumps(result))
        result = []


fileName = 'data_2238_time1626326022_default.jsonl	data_3511_time1626339364_default.jsonl	data_4781_time1626354542_default.jsonl	data_6039_time1626368878_default.jsonl	data_980_time1626313933_default.jsonl data_2239_time1626326051_default.jsonl	data_3514_time1626339402_default.jsonl	data_4782_time1626354547_default.jsonl	data_603_time1626310617_default.jsonl	data_981_time1626313954_default.jsonl data_223_time1626307376_default.jsonl	data_3515_time1626339422_default.jsonl	data_4783_time1626354575_default.jsonl	data_6040_time1626368890_default.jsonl	data_982_time1626313959_default.jsonl data_2240_time1626326062_default.jsonl	data_3516_time1626339430_default.jsonl	data_4784_time1626354584_default.jsonl	data_6041_time1626368901_default.jsonl	data_983_time1626313964_default.jsonl'

#fileName = 'data_2234_time1626325972_default.jsonl	data_3508_time1626339342_default.jsonl	data_4777_time1626354509_default.jsonl	data_6035_time1626368807_default.jsonl	data_978_time1626313916_default.jsonl data_2235_time1626325993_default.jsonl	data_3509_time1626339349_default.jsonl	data_4778_time1626354515_default.jsonl	data_6036_time1626368836_default.jsonl	data_979_time1626313924_default.jsonl data_2236_time1626326004_default.jsonl	data_350_time1626308452_default.jsonl	data_477_time1626309502_default.jsonl	data_6037_time1626368849_default.jsonl	data_97_time1626305498_default.jsonl data_2237_time1626326010_default.jsonl	data_3510_time1626339358_default.jsonl	data_4780_time1626354529_default.jsonl	data_6038_time1626368867_default.jsonl	data_97_time1626306385_default.jsonl'

#fileName = 'data_2230_time1626325908_default.jsonl	data_3504_time1626339291_default.jsonl	data_4773_time1626354468_default.jsonl	data_6031_time1626368767_default.jsonl	data_973_time1626313876_default.jsonl data_2231_time1626325918_default.jsonl	data_3505_time1626339301_default.jsonl	data_4774_time1626354474_default.jsonl	data_6032_time1626368773_default.jsonl	data_974_time1626313884_default.jsonl data_2232_time1626325923_default.jsonl	data_3506_time1626339309_default.jsonl	data_4775_time1626354481_default.jsonl	data_6033_time1626368780_default.jsonl	data_975_time1626313890_default.jsonl data_2233_time1626325939_default.jsonl	data_3507_time1626339335_default.jsonl	data_4776_time1626354496_default.jsonl	data_6034_time1626368800_default.jsonl	data_977_time1626313908_default.jsonl'

# fileName = 'data_2227_time1626325867_default.jsonl	data_34_time1626305882_default.jsonl	data_4769_time1626354426_default.jsonl	data_6028_time1626368739_default.jsonl	data_96_time1626306377_default.jsonl data_2228_time1626325881_default.jsonl	data_3500_time1626339246_default.jsonl	data_476_time1626309493_default.jsonl	data_6029_time1626368747_default.jsonl	data_970_time1626313846_default.jsonl data_2229_time1626325898_default.jsonl	data_3502_time1626339274_default.jsonl	data_4770_time1626354432_default.jsonl	data_602_time1626310608_default.jsonl	data_971_time1626313854_default.jsonl data_222_time1626307371_default.jsonl	data_3503_time1626339283_default.jsonl	data_4772_time1626354460_default.jsonl	data_6030_time1626368759_default.jsonl	data_972_time1626313864_default.jsonl'

#fileName = 'data_967_time1626313806_default.jsonl data_2224_time1626325823_default.jsonl	data_3498_time1626339218_default.jsonl	data_4766_time1626354382_default.jsonl	data_6025_time1626368694_default.jsonl	data_968_time1626313826_default.jsonl data_2225_time1626325841_default.jsonl	data_349_time1626308443_default.jsonl	data_4767_time1626354401_default.jsonl	data_6026_time1626368704_default.jsonl	data_969_time1626313829_default.jsonl data_2226_time1626325857_default.jsonl	data_34_time1626305001_default.jsonl	data_4768_time1626354412_default.jsonl	data_6027_time1626368717_default.jsonl	data_96_time1626305490_default.jsonl data_2227_time1626325867_default.jsonl	data_34_time1626305882_default.jsonl	data_4769_time1626354426_default.jsonl	data_6028_time1626368739_default.jsonl	data_96_time1626306377_default.jsonl '

# fileName =  'data_2241_time1626326067_default.jsonl	data_3517_time1626339439_default.jsonl	data_4785_time1626354594_default.jsonl	data_6042_time1626368924_default.jsonl	data_984_time1626313975_default.jsonl data_2242_time1626326083_default.jsonl	data_3518_time1626339452_default.jsonl	data_4786_time1626354600_default.jsonl	data_6043_time1626368936_default.jsonl	data_985_time1626313982_default.jsonl data_2243_time1626326100_default.jsonl	data_3519_time1626339463_default.jsonl	data_4787_time1626354612_default.jsonl	data_6044_time1626368950_default.jsonl	data_986_time1626313989_default.jsonl data_2244_time1626326101_default.jsonl	data_351_time1626308457_default.jsonl	data_4788_time1626354631_default.jsonl	data_6045_time1626368972_default.jsonl	data_987_time1626313994_default.jsonl'

#fileName = 'data_2245_time1626326109_default.jsonl	data_3520_time1626339473_default.jsonl	data_4789_time1626354664_default.jsonl	data_6046_time1626368986_default.jsonl	data_988_time1626314004_default.jsonl data_2246_time1626326114_default.jsonl	data_3521_time1626339483_default.jsonl	data_478_time1626309510_default.jsonl	data_6047_time1626369005_default.jsonl	data_989_time1626314013_default.jsonl data_2247_time1626326121_default.jsonl	data_3522_time1626339493_default.jsonl	data_4790_time1626354680_default.jsonl	data_6048_time1626369016_default.jsonl	data_98_time1626305503_default.jsonl data_2248_time1626326130_default.jsonl	data_3523_time1626339507_default.jsonl	data_4791_time1626354692_default.jsonl	data_6049_time1626369030_default.jsonl	data_98_time1626306389_default.jsonl'

# fileName = 'data_2249_time1626326141_default.jsonl	data_3524_time1626339514_default.jsonl	data_4792_time1626354699_default.jsonl	data_604_time1626310624_default.jsonl	data_991_time1626314024_default.jsonl data_224_time1626307389_default.jsonl	data_3525_time1626339521_default.jsonl	data_4793_time1626354722_default.jsonl	data_6050_time1626369043_default.jsonl	data_992_time1626314033_default.jsonl data_2250_time1626326151_default.jsonl	data_3526_time1626339531_default.jsonl	data_4794_time1626354742_default.jsonl	data_6051_time1626369049_default.jsonl	data_993_time1626314040_default.jsonl data_2251_time1626326158_default.jsonl	data_3527_time1626339546_default.jsonl	data_4795_time1626354767_default.jsonl	data_6052_time1626369063_default.jsonl	data_994_time1626314044_default.jsonl'

#fileName = 'data_2252_time1626326170_default.jsonl	data_3528_time1626339555_default.jsonl	data_4797_time1626354804_default.jsonl	data_6053_time1626369069_default.jsonl	data_995_time1626314052_default.jsonl data_2253_time1626326231_default.jsonl	data_3529_time1626339573_default.jsonl	data_4798_time1626354818_default.jsonl	data_6054_time1626369073_default.jsonl	data_996_time1626314060_default.jsonl data_2254_time1626326234_default.jsonl	data_352_time1626308470_default.jsonl	data_4799_time1626354827_default.jsonl	data_6055_time1626369073_default.jsonl	data_997_time1626314077_default.jsonl data_2255_time1626326245_default.jsonl	data_3530_time1626339584_default.jsonl	data_479_time1626309516_default.jsonl	data_6056_time1626369074_default.jsonl	data_998_time1626314084_default.jsonl'

# fileName = 'data_2256_time1626326246_default.jsonl	data_3531_time1626339593_default.jsonl	data_47_time1626305096_default.jsonl	data_6057_time1626369080_default.jsonl	data_999_time1626314090_default.jsonl data_2257_time1626326246_default.jsonl	data_3532_time1626339601_default.jsonl	data_47_time1626305979_default.jsonl	data_6058_time1626369103_default.jsonl	data_99_time1626306395_default.jsonl data_2258_time1626326247_default.jsonl	data_3533_time1626339612_default.jsonl	data_4800_time1626354838_default.jsonl	data_6059_time1626369113_default.jsonl	data_9_time1626304842_default.jsonl data_2259_time1626326266_default.jsonl	data_3534_time1626339620_default.jsonl	data_4801_time1626354848_default.jsonl	data_605_time1626310643_default.jsonl	data_9_time1626305722_default.jsonl'
#fileName = 'data_1007_time1626314174_default.jsonl	data_2269_time1626326342_default.jsonl	data_3544_time1626339726_default.jsonl	data_4811_time1626354950_default.jsonl	data_606_time1626310648_default.jsonl data_1008_time1626314182_default.jsonl	data_226_time1626307417_default.jsonl	data_3545_time1626339737_default.jsonl	data_4812_time1626354964_default.jsonl	data_6070_time1626369246_default.jsonl data_1009_time1626314192_default.jsonl	data_2270_time1626326347_default.jsonl	data_3546_time1626339755_default.jsonl	data_4814_time1626355007_default.jsonl	data_6071_time1626369252_default.jsonl data_100_time1626306399_default.jsonl	data_2271_time1626326364_default.jsonl	data_3547_time1626339760_default.jsonl	data_4815_time1626355014_default.jsonl	data_6072_time1626369263_default.jsonl data_1010_time1626314201_default.jsonl	data_2272_time1626326378_default.jsonl	data_3548_time1626339768_default.jsonl	data_4816_time1626355025_default.jsonl	data_6073_time1626369279_default.jsonl data_1011_time1626314207_default.jsonl	data_2273_time1626326389_default.jsonl	data_3549_time1626339795_default.jsonl	data_4817_time1626355069_default.jsonl	data_6074_time1626369288_default.jsonl data_1012_time1626314213_default.jsonl	data_2275_time1626326410_default.jsonl	data_354_time1626308496_default.jsonl	data_4818_time1626355078_default.jsonl	data_6075_time1626369295_default.jsonl data_1013_time1626314215_default.jsonl	data_2276_time1626326429_default.jsonl	data_3550_time1626339805_default.jsonl	data_4819_time1626355082_default.jsonl	data_6076_time1626369307_default.jsonl'

#fileName = 'data_1007_time1626314174_default.jsonl	data_2269_time1626326342_default.jsonl	data_3544_time1626339726_default.jsonl	data_4811_time1626354950_default.jsonl	data_606_time1626310648_default.jsonl data_1008_time1626314182_default.jsonl	data_226_time1626307417_default.jsonl	data_3545_time1626339737_default.jsonl	data_4812_time1626354964_default.jsonl	data_6070_time1626369246_default.jsonl data_1009_time1626314192_default.jsonl	data_2270_time1626326347_default.jsonl	data_3546_time1626339755_default.jsonl	data_4814_time1626355007_default.jsonl	data_6071_time1626369252_default.jsonl data_100_time1626306399_default.jsonl	data_2271_time1626326364_default.jsonl	data_3547_time1626339760_default.jsonl	data_4815_time1626355014_default.jsonl	data_6072_time1626369263_default.jsonl'

# fileName = 'data_1002_time1626314127_default.jsonl	data_2265_time1626326311_default.jsonl	data_3540_time1626339692_default.jsonl	data_4808_time1626354901_default.jsonl	data_6066_time1626369202_default.jsonl data_1003_time1626314134_default.jsonl	data_2266_time1626326318_default.jsonl	data_3541_time1626339699_default.jsonl	data_4809_time1626354905_default.jsonl	data_6067_time1626369225_default.jsonl data_1005_time1626314152_default.jsonl	data_2267_time1626326328_default.jsonl	data_3542_time1626339710_default.jsonl	data_480_time1626309525_default.jsonl	data_6068_time1626369233_default.jsonl data_1006_time1626314163_default.jsonl	data_2268_time1626326338_default.jsonl	data_3543_time1626339718_default.jsonl	data_4810_time1626354920_default.jsonl	data_6069_time1626369238_default.jsonl'

# fileName = 'data_0_time1626304785_default.jsonl	data_2260_time1626326274_default.jsonl	data_3536_time1626339645_default.jsonl	data_4803_time1626354851_default.jsonl	data_6061_time1626369132_default.jsonl data_0_time1626305664_default.jsonl	data_2261_time1626326281_default.jsonl	data_3538_time1626339670_default.jsonl	data_4805_time1626354886_default.jsonl	data_6062_time1626369154_default.jsonl data_1000_time1626314094_default.jsonl	data_2262_time1626326288_default.jsonl	data_3539_time1626339686_default.jsonl	data_4806_time1626354890_default.jsonl	data_6063_time1626369161_default.jsonl data_1001_time1626314108_default.jsonl	data_2264_time1626326303_default.jsonl	data_353_time1626308491_default.jsonl	data_4807_time1626354895_default.jsonl	data_6064_time1626369169_default.jsonl'


fileList = fileName.split()
basedir = './pyjsons/'

for fileName in fileList:
    print(fileName)
    filePath = './Clippy_Parsed_Functions_1/'+fileName
    if not os.path.exists(filePath):
        os.makedirs(filePath)
    with open(basedir+fileName) as file:
        count = 0
        for line in file:
            jsonObj = json.loads(line)
            count += 1
            with open(filePath+"/"+str(count)+".py", "w") as wfile:
                # Writing data to a file
                wfile.write(jsonObj['text'])
        analyze(filePath, fileName)
        # shutil.rmtree(filePath, ignore_errors=True)
        # print("Deleted '%s' directory successfully" % filePath)
