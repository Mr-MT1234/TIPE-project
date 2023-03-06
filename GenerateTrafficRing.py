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
    additionals = config.createElement('additionals')

    net = config.createElement('net-file')
    net.setAttribute('value', 'ring.net.xml')

    rerouters = config.createElement('additional-files')
    rerouters.setAttribute('value','ring.rerou.xml')

    input.appendChild(net)
    additionals.appendChild(rerouters)
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







