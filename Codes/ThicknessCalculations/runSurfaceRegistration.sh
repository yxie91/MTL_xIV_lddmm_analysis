
# Examples:
# Codes/ThicknessCalculations/runSurfaceRegistration.sh -s 1 -i input_path -o output_path -f all 

while getopts s:i:f:o: flag
do
    case "${flag}" in
        s) step=${OPTARG};; # 1 or 0 to be flipping  
        i) inputDir=${OPTARG};; # indicate directory to get input top and bottom with 
        f) fil=${OPTARG};; # indicate file prefix to use (if all, then grab all occurrences of top and bottom)
        o) outPath=${OPTARG};; # output path to which to write template and evolution (will create subdirectory with fil pref)
    esac
done



if [[ $fil == "all" ]]; then
    fils=$(find $inputDir | grep top | grep vtk | grep -v Hold)
else
    fils=$(find $inputDir | grep top | grep vtk | grep -v Hold | grep $fil)
fi

echo $fils

for t in ${fils[*]}; do
    b=${t%top.vtk}bottom.vtk
    bt=$(basename $t)
    od=$outPath${bt%_top.vtk}/
    mkdir -m 777 $od

    python3 Codes/ThicknessCalculations/py-lddmm1/SurfaceRegistration_originalEdit.py $t $b $od $step
    python3  -c "from sys import path as sys_path; sys_path.append('Codes/ThicknessCalculations/vtkFunctions.py'); import vtkFunctions as vt; vt.getThickness('${od}Template.vtk','${od}evolution20.vtk','${od}thickness.vtk');quit()"
done
