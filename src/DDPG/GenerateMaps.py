from xml.dom import minidom
import math
import subprocess
import os


def createRingFiles(Radius, NodeNumber ,EdgeResolution, outputDir):
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    nodes = minidom.Document()
    root = nodes.createElement('nodes')
    nodes.appendChild(root)
    
    en = complex(math.cos(2*math.pi/NodeNumber),math.sin(2*math.pi/NodeNumber))
    pos = Radius
    for i in range(NodeNumber):
        node = nodes.createElement('node')
        node.setAttribute('id'  ,'n' + str(i))
        node.setAttribute('x'   ,str(pos.real))
        node.setAttribute('y'   ,str(pos.imag))
        node.setAttribute('type','priority')
        pos *= en

        root.appendChild(node)

    f = open(outputDir + 'ring.nodes.xml','w')
    f.write(nodes.toprettyxml())
    f.close()

    del nodes

    edges = minidom.Document()
    root = edges.createElement('edges')
    edges.appendChild(root)

    ee = complex(math.cos(2*math.pi/(EdgeResolution*NodeNumber)),
                math.sin(2*math.pi/(EdgeResolution*NodeNumber)))
    beginPos = Radius
    for i in range(NodeNumber):
        edge = edges.createElement('edge')
        edge.setAttribute('id'      ,'e' + str(i))
        edge.setAttribute('from'    ,'n' + str(i))
        edge.setAttribute('to'      ,'n' + str((i+1)%(NodeNumber)))
        edge.setAttribute('priority','1')
        edge.setAttribute('speed'   , '30.0')

        currentPos = beginPos
        shape = ' {},{}'.format(currentPos.real,currentPos.imag)
        for j in range(1,EdgeResolution+1):
            currentPos *= ee
            shape += ' {},{}'.format(currentPos.real,currentPos.imag)
        beginPos *= en

        edge.setAttribute('shape',shape)
        root.appendChild(edge)

    f = open(outputDir + 'ring.edges.xml','w')
    f.write(edges.toprettyxml())
    f.close()

    del edges

    rerouters = minidom.Document()
    root = rerouters.createElement('additionals')
    rerouters.appendChild(root)

    for i in range(NodeNumber):
        rerouter = rerouters.createElement('rerouter')
        rerouter.setAttribute('id'   , 'rr'+str(i))
        rerouter.setAttribute('edges', 'e' +str(i))

        interval = rerouters.createElement('interval')
        interval.setAttribute('end','1e9')

        dest = rerouters.createElement('destProbReroute')
        dest.setAttribute('id','e'+str((i+1)%(NodeNumber)))

        interval.appendChild(dest)
        rerouter.appendChild(interval)
        root.appendChild(rerouter)

    f = open(outputDir + 'ring.rerou.xml','w')
    f.write(rerouters.toprettyxml())
    f.close()

    del rerouters

    config = minidom.Document()
    root = config.createElement('configuration')
    config.appendChild(root)

    input = config.createElement('input')

    net = config.createElement('net-file')
    net.setAttribute('value', 'ring.net.xml')

    additionals = config.createElement('additionals')
    rerouters = config.createElement('additional-files')
    rerouters.setAttribute('value','ring.rerou.xml')
    additionals.appendChild(rerouters)

    input.appendChild(net)
    root.appendChild(input)
    root.appendChild(additionals)

    f = open(outputDir + 'ring.sumocfg','w')
    f.write(config.toprettyxml())
    f.close()

    del config

    subprocess.run(['netconvert',
    '-n', outputDir + 'ring.nodes.xml',
    '-e', outputDir +'ring.edges.xml',
    '-o', outputDir + 'ring.net.xml'],
    shell=True)


def createIntersectionFiles(preLenght, postLenght, outputDir):
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    nodes = minidom.Document()
    root = nodes.createElement('nodes')
    nodes.appendChild(root)
    
    nodeCenter = nodes.createElement('node')
    nodeCenter.setAttribute('id'  ,'nCenter')
    nodeCenter.setAttribute('x'   ,str(0))
    nodeCenter.setAttribute('y'   ,str(0))
    nodeCenter.setAttribute('type','priority')
    
    nodeLeft = nodes.createElement('node')
    nodeLeft.setAttribute('id'  ,'nLeft')
    nodeLeft.setAttribute('x'   ,str(-preLenght))
    nodeLeft.setAttribute('y'   ,str(0))
    nodeLeft.setAttribute('type','priority')
    
    nodeRight = nodes.createElement('node')
    nodeRight.setAttribute('id'  ,'nRight')
    nodeRight.setAttribute('x'   ,str(+preLenght))
    nodeRight.setAttribute('y'   ,str(0))
    nodeRight.setAttribute('type','priority')
    
    nodePost = nodes.createElement('node')
    nodePost.setAttribute('id'  ,'nPost')
    nodePost.setAttribute('x'   ,str(0))
    nodePost.setAttribute('y'   ,str(postLenght))
    nodePost.setAttribute('type','priority')
    
    nodeLeftEnt = nodes.createElement('node')
    nodeLeftEnt.setAttribute('id'  ,'nLeftEnt')
    nodeLeftEnt.setAttribute('x'   ,str(-preLenght - 100))
    nodeLeftEnt.setAttribute('y'   ,str(0))
    nodeLeftEnt.setAttribute('type','priority')
    
    nodeRightEnt = nodes.createElement('node')
    nodeRightEnt.setAttribute('id'  ,'nRightEnt')
    nodeRightEnt.setAttribute('x'   ,str(+preLenght + 100))
    nodeRightEnt.setAttribute('y'   ,str(0))
    nodeRightEnt.setAttribute('type','priority')
    
    nodePostExt = nodes.createElement('node')
    nodePostExt.setAttribute('id'  ,'nPostExt')
    nodePostExt.setAttribute('x'   ,str(0))
    nodePostExt.setAttribute('y'   ,str(postLenght + 100))
    nodePostExt.setAttribute('type','priority')
    
    root.appendChild(nodeCenter)
    root.appendChild(nodeRight)
    root.appendChild(nodeLeft)
    root.appendChild(nodePost)
    root.appendChild(nodeLeftEnt)
    root.appendChild(nodeRightEnt)
    root.appendChild(nodePostExt)

    f = open(outputDir + '/intersection.nodes.xml','w')
    f.write(nodes.toprettyxml())
    f.close()

    del nodes

    edges = minidom.Document()
    root = edges.createElement('edges')
    edges.appendChild(root)

    
    edgeLL = edges.createElement('edge')
    edgeLL.setAttribute('id'      ,'eLL')
    edgeLL.setAttribute('from'    ,'nLeftEnt')
    edgeLL.setAttribute('to'      ,'nLeft')
    edgeLL.setAttribute('priority','1')
    edgeLL.setAttribute('speed'   , '30.0')
    root.appendChild(edgeLL)
    
    edgeLC = edges.createElement('edge')
    edgeLC.setAttribute('id'      ,'eLC')
    edgeLC.setAttribute('from'    ,'nLeft')
    edgeLC.setAttribute('to'      ,'nCenter')
    edgeLC.setAttribute('priority','1')
    edgeLC.setAttribute('speed'   , '30.0')
    root.appendChild(edgeLC)
    
    edgeRR = edges.createElement('edge')
    edgeRR.setAttribute('id'      ,'eRR')
    edgeRR.setAttribute('from'    ,'nRightEnt')
    edgeRR.setAttribute('to'      ,'nRight')
    edgeRR.setAttribute('priority','1')
    edgeRR.setAttribute('speed'   , '30.0')
    root.appendChild(edgeRR)
    
    edgeRC = edges.createElement('edge')
    edgeRC.setAttribute('id'      ,'eRC')
    edgeRC.setAttribute('from'    ,'nRight')
    edgeRC.setAttribute('to'      ,'nCenter')
    edgeRC.setAttribute('priority','1')
    edgeRC.setAttribute('speed'   , '30.0')
    root.appendChild(edgeRC)
    
    edgeCP = edges.createElement('edge')
    edgeCP.setAttribute('id'      ,'eCP')
    edgeCP.setAttribute('from'    ,'nCenter')
    edgeCP.setAttribute('to'      ,'nPost')
    edgeCP.setAttribute('priority','1')
    edgeCP.setAttribute('speed'   , '30.0')
    root.appendChild(edgeCP)
    
    edgePP = edges.createElement('edge')
    edgePP.setAttribute('id'      ,'ePP')
    edgePP.setAttribute('from'    ,'nPost')
    edgePP.setAttribute('to'      ,'nPostExt')
    edgePP.setAttribute('priority','1')
    edgePP.setAttribute('speed'   , '30.0')
    root.appendChild(edgePP)
    
    edgePL = edges.createElement('edge')
    edgePL.setAttribute('id'      ,'ePL')
    edgePL.setAttribute('from'    ,'nPost')
    edgePL.setAttribute('to'      ,'nLeft')
    edgePL.setAttribute('priority','1')
    edgePL.setAttribute('speed'   , '30.0')
    edgePL.setAttribute('shape'   , f'0,{postLenght} {-preLenght},{postLenght} {-preLenght},0')
    root.appendChild(edgePL)
    
    edgePR = edges.createElement('edge')
    edgePR.setAttribute('id'      ,'ePR')
    edgePR.setAttribute('from'    ,'nPost')
    edgePR.setAttribute('to'      ,'nRight')
    edgePR.setAttribute('priority','1')
    edgePR.setAttribute('speed'   , '30.0')
    edgePR.setAttribute('shape'   , f'0,{postLenght} {preLenght},{postLenght} {preLenght},0')
    root.appendChild(edgePR)

    f = open(outputDir + '/intersection.edges.xml','w')
    f.write(edges.toprettyxml())
    f.close()

    del edges
    
    rerouters = minidom.Document()
    root = rerouters.createElement('additionals')
    rerouters.appendChild(root)
    rerouter = rerouters.createElement('rerouter')
    rerouter.setAttribute('id'   , 'rrr')
    rerouter.setAttribute('edges', 'ePR')

    interval = rerouters.createElement('interval')
    interval.setAttribute('end','1e9')

    dest = rerouters.createElement('destProbReroute')
    dest.setAttribute('id','eCP')

    interval.appendChild(dest)
    rerouter.appendChild(interval)
    root.appendChild(rerouter)
    
    rerouter2 = rerouters.createElement('rerouter')
    rerouter2.setAttribute('id'   , 'rrl')
    rerouter2.setAttribute('edges', 'ePL')

    interval2 = rerouters.createElement('interval')
    interval2.setAttribute('end','1e9')

    dest2 = rerouters.createElement('destProbReroute')
    dest2.setAttribute('id','eCP')

    interval2.appendChild(dest2)
    rerouter2.appendChild(interval2)
    root.appendChild(rerouter2)
    
    f = open(outputDir + '/intersection.rerou.xml','w')
    f.write(rerouters.toprettyxml())
    f.close()

    del rerouters
    
    config = minidom.Document()
    root = config.createElement('configuration')
    config.appendChild(root)

    input = config.createElement('input')

    net = config.createElement('net-file')
    net.setAttribute('value', 'intersection.net.xml')
    
    additionals = config.createElement('additionals')
    rerouters = config.createElement('additional-files')
    rerouters.setAttribute('value','intersection.rerou.xml')
    additionals.appendChild(rerouters)

    input.appendChild(net)
    root.appendChild(input)
    root.appendChild(additionals)

    f = open(outputDir + '/intersection.sumocfg','w')
    f.write(config.toprettyxml())
    f.close()

    del config

    subprocess.run(['netconvert',
    '-n', outputDir + '/intersection.nodes.xml',
    '-e', outputDir + '/intersection.edges.xml',
    '-o', outputDir + '/intersection.net.xml'],
    shell=True)







