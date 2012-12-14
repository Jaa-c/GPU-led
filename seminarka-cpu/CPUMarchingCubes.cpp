#include <iostream>

#include "CPUMarchingCubes.h"

CPUMarchingCubes::CPUMarchingCubes(const int dataWidth, const int dataHeight, const int dataDepth) {
	this->dataWidth = dataWidth;
	this->dataHeight = dataHeight;
	this->dataDepth = dataDepth;
}

/**
 * Marching cubes pro celou møížku
 *
 */
void CPUMarchingCubes::vMarchingCubes(Voxel* data, const int dataCount)
{
	this->data = data;

	int iX, iY, iZ;
    for(iX = 0; iX < dataWidth; iX++)
		for(iY = 0; iY < dataHeight; iY++)
			for(iZ = 0; iZ < dataDepth; iZ++) {
				vMarchCube(iX, iY, iZ, dataCount);
			}
}



/**
 * Marching cubes na jedný kostce (8 voxelù)
 *
 * použit základ kódu z http://www.siafoo.net/snippet/100
 *
 */
 void CPUMarchingCubes::vMarchCube(const int fX, const  int fY, const int fZ, const int dataCount, const GLfloat fScale) {
        GLint iCorner, iVertex, iVertexTest, iEdge, iTriangle, iFlagIndex, iEdgeFlags;
		GLfloat fOffset;
		Voxel* afCubeValue[8];
		GLvector asEdgeVertex[12];
		GLvector asEdgeNorm[12];

		//GLfloat triangleNet[dataCount*3];
		//int triangleNetIndex;
        
        //Make a local copy of the values at the cube's corners
        for(iVertex = 0; iVertex < 8; iVertex++)
        {
			if( fX + a2fVertexOffset[iVertex][0] < 0 || fX + a2fVertexOffset[iVertex][0] >= dataWidth ||
				fY + a2fVertexOffset[iVertex][1] < 0 || fY + a2fVertexOffset[iVertex][1] >= dataHeight ||
				fZ + a2fVertexOffset[iVertex][2] < 0 || fZ + a2fVertexOffset[iVertex][2] >= dataDepth) {
					 afCubeValue[iVertex] = NULL;
					 continue;
			}
			afCubeValue[iVertex] =  &data[DATA_INDEX(fX + a2fVertexOffset[iVertex][0],
										 fY + a2fVertexOffset[iVertex][1],
										 fZ + a2fVertexOffset[iVertex][2])];
        }

        //Find which vertices are inside of the surface and which are outside
        iFlagIndex = 0;
        for(iVertexTest = 0; iVertexTest < 8; iVertexTest++)
        {
			if(afCubeValue[iVertexTest] == NULL) {
				continue;
			}
			if(afCubeValue[iVertexTest]->status == ICE) 
				iFlagIndex |= 1 << iVertexTest;
        }

        //Find which edges are intersected by the surface
        iEdgeFlags = aiCubeEdgeFlags[iFlagIndex];

        //If the cube is entirely inside or outside of the surface, then there will be no intersections
        if(iEdgeFlags == 0) 
			return;
        
		
        //Find the point of intersection of the surface with each edge
        //Then find the normal to the surface at those points
        for(iEdge = 0; iEdge < 12; iEdge++)
        {
            //if there is an intersection on this edge
            if(iEdgeFlags & (1<<iEdge))
            {
				//TODO!!!
				fOffset = 0.5f;

				float* v = data[DATA_INDEX(fX, fY, fZ)].position;

                asEdgeVertex[iEdge].fX = v[0] + (a2fVertexOffset[ a2iEdgeConnection[iEdge][0] ][0] + fOffset * a2fEdgeDirection[iEdge][0]) *fScale;
                asEdgeVertex[iEdge].fY = v[1] + (a2fVertexOffset[ a2iEdgeConnection[iEdge][0] ][1] + fOffset * a2fEdgeDirection[iEdge][1]) *fScale;
                asEdgeVertex[iEdge].fZ = v[2] + (a2fVertexOffset[ a2iEdgeConnection[iEdge][0] ][2] + fOffset * a2fEdgeDirection[iEdge][2]) *fScale;
                //vGetNormal(asEdgeNorm[iEdge], asEdgeVertex[iEdge].fX, asEdgeVertex[iEdge].fY, asEdgeVertex[iEdge].fZ);
            }
        }


        //Draw the triangles that were found.  There can be up to five per cube
        for(iTriangle = 0; iTriangle < 5; iTriangle++)
        {
            if(a2iTriangleConnectionTable[iFlagIndex][3*iTriangle] < 0)
                    break;
			
            for(iCorner = 0; iCorner < 3; iCorner++)
            {
                iVertex = a2iTriangleConnectionTable[iFlagIndex][3*iTriangle+iCorner];

                //glNormal3f(normal.fX, normal.fY, normal.fZ);
                glVertex3f(asEdgeVertex[iVertex].fX, asEdgeVertex[iVertex].fY, asEdgeVertex[iVertex].fZ); //TODO
				
				//triangleNet[triangleNetIndex] = asEdgeVertex[iVertex].fX;
				//triangleNet[triangleNetIndex] = asEdgeVertex[iVertex].fY;
				//triangleNet[triangleNetIndex] = asEdgeVertex[iVertex].fZ;

            }
        }
}