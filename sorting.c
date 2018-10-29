#include<stdio.h>
int main() {
 long N,X[100],j,i,S[100],temp;
    scanf("%d",&N);
    for(i=0;i<N;i++){
        scanf("%d",&X[i]);
    }
	for(j=0;j<N;j++){
    for(i=0;i<N-j-1;i++){
      if(X[i]>X[i+1]){
      temp=X[i];
      X[i]=X[i+1];
      X[i+1]=temp;}
    }}
    
     for(i=0;i<N;i++){
        printf("%d ",X[i]);
    }
    printf("\n----------Sorted--------------");
    
}
