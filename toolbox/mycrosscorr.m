function crosscorrcoeff = mycrosscorr(x,y,lagx)
%
crosscorrcoeff=[];
for i=0:lagx
crosscorrcoeff=[crosscorrcoeff corr(x(1:end-i),y(i+1:end))];
end
plot(0:lagx,crosscorrcoeff,'.-')
end

