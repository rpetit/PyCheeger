static void l1ballproj_Condat(datatype* y, datatype* x, const unsigned int length, const double a) {
	if (a<=0.0) {
		if (a==0.0) memset(x,0,length*sizeof(datatype));
		return;
	}
	datatype*	aux = (x==y ? (datatype*)malloc(length*sizeof(datatype)) : x);
	int		auxlength=1;
	int		auxlengthold=-1;
	double	tau=(*aux=(*y>=0.0 ? *y : -*y))-a;
	int 	i=1;
	for (; i<length; i++)
		if (y[i]>0.0) {
			if (y[i]>tau) {
				if ((tau+=((aux[auxlength]=y[i])-tau)/(auxlength-auxlengthold))
				<=y[i]-a) {
					tau=y[i]-a;
					auxlengthold=auxlength-1;
				}
				auxlength++;
			}
		} else if (y[i]!=0.0) {
			if (-y[i]>tau) {
				if ((tau+=((aux[auxlength]=-y[i])-tau)/(auxlength-auxlengthold))
				<=aux[auxlength]-a) {
					tau=aux[auxlength]-a;
					auxlengthold=auxlength-1;
				}
				auxlength++;
			}
		}
	if (tau<=0) {	/* y is in the l1 ball => x=y */
		if (x!=y) memcpy(x,y,length*sizeof(datatype));
		else free(aux);
	} else {
		datatype*  aux0=aux;
		if (auxlengthold>=0) {
			auxlength-=++auxlengthold;
			aux+=auxlengthold;
			while (--auxlengthold>=0)
				if (aux0[auxlengthold]>tau)
					tau+=((*(--aux)=aux0[auxlengthold])-tau)/(++auxlength);
		}
		do {
			auxlengthold=auxlength-1;
			for (i=auxlength=0; i<=auxlengthold; i++)
				if (aux[i]>tau)
					aux[auxlength++]=aux[i];
				else
					tau+=(tau-aux[i])/(auxlengthold-i+auxlength);
		} while (auxlength<=auxlengthold);
		for (i=0; i<length; i++)
			x[i]=(y[i]-tau>0.0 ? y[i]-tau : (y[i]+tau<0.0 ? y[i]+tau : 0.0));
		if (x==y) free(aux0);
	}
}