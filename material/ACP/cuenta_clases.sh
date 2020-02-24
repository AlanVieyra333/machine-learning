rm -rf clases
awk -F, '{print $5}' iris.csv > clases
awk -F, '{ if( $1==0 ) a=a+1 } END{print a}' clases
awk -F, '{ if( $1==1 ) a=a+1 } END{print a}' clases
awk -F, '{ if( $1==2 ) a=a+1 } END{print a}' clases
