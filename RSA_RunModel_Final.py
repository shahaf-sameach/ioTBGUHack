
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis

from sklearn.metrics import (f1_score,recall_score,precision_score, classification_report,precision_recall_curve,average_precision_score)


# ## Load Data and remove string columns

# In[ ]:


RSA = pd.read_csv('Holdout_v1.csv').drop(["Unnamed: 0"],axis=1)


# In[ ]:


TOP_FI=['Length' ,'NonAlphaNumericToLength' ,'TLD_LEN' ,'NonAlphaNumeric' ,'DigitsToLength' ,'Dots' ,'Digits' ,'?s' ,'IsHTML' ,'www.1' ,'DigitsInSubDomain' ,'wordTwice' ,'IsEXE' ,'DigitsInTopLevelDomain' ,'StartWithWWW' ,'SP500WordsNotExact' ,'www' ,'No_Data.1' ,'Http' ,'bodydesign' ,'com.1' ,'whois.PublicDomainRegistry.com' ,'au' ,'com' ,'com.au' ,'whois.godaddy.com' ,'%s' ,'No_Data' ,'SP500WordsNotExactInTLD' ,'cn' ,'whois.networksolutions.com' ,'de' ,'de.1' ,'com.br' ,'blogspot' ,'whois.markmonitor.com' ,'br.1' ,'whois.ename.com' ,'cn.1' ,'net.1' ,'whois.tucows.com' ,'US' ,'missleadingWords' ,'other.1' ,'net' ,'id.1' ,'whois.enom.com' ,'ru' ,'jp.1' ,'whois.web.com' ,'paypal' ,'ru.1' ,'info.1' ,'JP' ,'CN' ,'org.1' ,'account' ,'blogspot.com' ,'other' ,'whois.namecheap.com' ,'apple' ,'SP500WordsExactInSubDomain' ,'co.1' ,'org' ,'IsFile' ,'info' ,'whois.paycenter.com.cn' ,'esy' ,'SP500WordsExact' ,'in.1' ,'cl' ,'SP500WordsNotExactInSubDomain' ,'whois.NameBright.com' ,'jp' ,'uk.1' ,'whois.wildwestdomains.com' ,'cl.1' ,'whois.ovh.com' ,'myjino' ,'whois.1and1.com' ,'nl' ,'online.1' ,'pl.1' ,'whois.hostinger.com' ,'in' ,'nl.1' ,'whois.register.com' ,'cf' ,'it.1' ,'whois.domain.com' ,'appleid' ,'whois.namesilo.com' ,'whois.name.com' ,'xyz' ,'cf.1' ,'xyz.1' ,'login' ,'club.1' ,'whois.gandi.net' ,'club' ,'edu.1' ,'grs-whois.hichina.com' ,'whois.discount-domain.com' ,'fr.1' ,'whois.bizcn.com' ,'ga' ,'whois.launchpad.com' ,'tk' ,'tk.1' ,'ga.1' ,'DE' ,'us.1' ,'top.1' ,'pl' ,'top' ,'cn.2' ,'es.1' ,'icloud' ,'fr' ,'whois.ascio.com' ,'whois.rrpproxy.net' ,'online' ,'IN' ,'000webhostapp' ,'it' ,'co.uk' ,'es' ,'ID' ,'whois.internet.bs' ,'blog' ,'ml.1' ,'ml' ,'update' ,'GB' ,'whois.onlinenic.com' ,'PA' ,'io.1' ,'gov.1' ,'ca.1' ,'whois.dreamhost.com' ,'tumblr' ,'pk.1' ,'ro' ,'ro.1' ,'co.id' ,'whois.isimtescil.net' ,'za' ,'ca' ,'us' ,'whois.bluehost.com' ,'cc.1' ,'co.za' ,'whois.corporatedomains.com' ,'Whois.bigrock.com' ,'m' ,'eu.1' ,'hol' ,'cc' ,'whois.google.com' ,'biz.1' ,'edu' ,'ua.1' ,'me.1' ,'eu' ,'signin' ,'co' ,'CA' ,'ie.1' ,'mx.1' ,'biz' ,'pe.1' ,'whois.west.cn' ,'my.1' ,'service' ,'whois.dynadot.com' ,'me' ,'support' ,'kr.1' ,'com.cn' ,'ES' ,'s' ,'ir' ,'xn' ,'ir.1' ,'at.1' ,'whois.publicdomainregistry.com' ,'sandwichdrip' ,'ar' ,'TR' ,'ng' ,'gq' ,'RU' ,'com.ar' ,'gq.1' ,'Austria' ,'ch.1' ,'gdn' ,'be.1' ,'whois.maff.com' ,'whois.meshdigital.com' ,'microsoft' ,'KR' ,'ch' ,'FR' ,'gdn.1' ,'IT' ,'secure' ,'mail' ,'co.in' ,'vn.1' ,'grs-whois.cndns.com' ,'gr.1' ,'shop' ,'whois.webnic.cc' ,'id' ,'win.1' ,'be' ,'pw.1' ,'se.1' ,'security' ,'e' ,'io' ,'i' ,'pw' ,'hu.1' ,'tr' ,'whois.ilovewww.com' ,'whois.advancedregistrar.com' ,'gr' ,'win' ,'whois.nicline.com' ,'system' ,'whois.eranet.com' ,'hu' ,'cdn' ,'com.mx' ,'whois.reg.com' ,'se' ,'dk.1' ,'at' ,'tech.1' ,'co.jp' ,'whois.yoursrs.com' ,'whois.1api.net' ,'BR' ,'whois.ihs.com.tr' ,'wordpress' ,'tw.1' ,'cz.1' ,'VN' ,'free' ,'dk' ,'sophosxl' ,'facebook' ,'5gbfree' ,'cloudfront' ,'sg.1' ,'verification' ,'site.1' ,'pro.1' ,'com.ng' ,'cz' ,'whois.psi-usa.info' ,'il' ,'tv.1' ,'ve' ,'nz.1' ,'whois.jprs.jp' ,'MX' ,'cloudfront.net' ,'whois.evonames.com' ,'ie' ,'com.my' ,'blogspot.ie' ,'website.1' ,'cutestat' ,'whois.safenames.net' ,'news.1' ,'whois.lexsynergy.com' ,'vn' ,'verify' ,'pt.1' ,'whois.domainpeople.com' ,'com.tr' ,'web' ,'a' ,'en' ,'whois-generic.marcaria.com' ,'app' ,'pp' ,'sitey' ,'NL' ,'whois.register.it' ,'no.1' ,'mobile' ,'com.ua' ,'website' ,'ac' ,'mobi.1' ,'whois.srsplus.com' ,'co.kr' ,'whois.resellercamp.com' ,'test' ,'ae.1' ,'whois.moniker.com' ,'16mb' ,'whois.directnic.com' ,'whois.55hl.com' ,'wikia' ,'co.il' ,'whois.35.com' ,'whois.22.cn' ,'ebay' ,'dev' ,'download.1' ,'rs.1' ,'link.1' ,'beget' ,'sk.1' ,'pagesstudy' ,'whois.nicproxy.com' ,'pt' ,'whois.syrahost.com' ,'site' ,'weebly' ,'hub' ,'tech' ,'whois.fastdomain.com' ,'center.1' ,'store.1' ,'hk.1' ,'on' ,'no' ,'SP500WordsExactInTLD' ,'pk' ,'PK' ,'whois.hostmonster.com' ,'tv' ,'mobi' ,'whois.synergywholesale.com' ,'NG' ,'lt' ,'com.pl' ,'whois.gabia.com' ,'whois.registrar.amazon.com' ,'PE' ,'bb' ,'lt.1' ,'whois.netregistry.com.au' ,'com.tw' ,'whois.dns.com.cn' ,'nimp' ,'github' ,'ua' ,'co.nz' ,'tripod' ,'whois.net4domains.com' ,'github.io' ,'com.pk' ,'whois.antagus.de' ,'BD' ,'com.sg' ,'ph.1' ,'sk' ,'pages' ,'whois.bookmyname.com' ,'services.1' ,'whois.registrar.eu' ,'google' ,'IR' ,'uy' ,'by.1' ,'bid.1' ,'th' ,'ae' ,'mx' ,'mdmag' ,'by' ,'fc2' ,'abuse' ,'pro' ,'whois.registrygate.com' ,'alert' ,'usaa' ,'help' ,'page' ,'whois.nic.ru' ,'whois.uniregistrar.net' ,'bid' ,'com.ve' ,'fi.1' ,'51qqxx' ,'disqus' ,'home' ,'hotel' ,'whois.scip.es' ,'api' ,'whois.joker.com' ,'org.uk' ,'whois.extranetdeclientes.com' ,'whois.gkg.net' ,'static' ,'the' ,'ZA' ,'new' ,'whois.names4ever.com' ,'dropbox' ,'si.1' ,'abqrm' ,'livejournal' ,'whois.neubox.com' ,'accounts' ,'cloud.1' ,'twomini' ,'hatenablog' ,'webapps' ,'whois.tldregistrarsolutions.com' ,'group' ,'whois.dattatec.com' ,'go' ,'sa' ,'fb' ,'amazon' ,'Thailand' ,'1.2' ,'kz.1' ,'net.au' ,'CH' ,'whois.liquidnetlimited.co.uk' ,'kz' ,'altervista' ,'loan' ,'fi' ,'loan.1' ,'email.1' ,'PL' ,'link' ,'IE' ,'gov.cn' ,'la.1' ,'AT' ,'media.1' ,'3eeweb' ,'whois.softlayer.com' ,'vk' ,'ge.1' ,'china' ,'webcindario' ,'bt' ,'chase' ,'UA' ,'for' ,'nf' ,'HK' ,'down' ,'c' ,'life.1' ,'access' ,'whois.reg2c.com' ,'bg.1' ,'3' ,'error' ,'iphone' ,'su' ,'incredible' ,'site1' ,'your' ,'art' ,'downyouxi' ,'whois.omnis.com' ,'lk.1' ,'MY' ,'url_' ,'com.vn' ,'su.1' ,'whois.net-chinese.com.tw' ,'readthedocs' ,'asia.1' ,'tekblue' ,'si' ,'ge' ,'intend' ,'click.1' ,'ws.1' ,'space.1' ,'96' ,'or' ,'health' ,'whois.cdmon.com' ,'hr.1' ,'KE' ,'our' ,'nf.1' ,'lk' ,'com.pe' ,'pubnub' ,'bankofamerica' ,'demo' ,'b' ,'d' ,'al.1' ,'whois.corehub.net' ,'org.br' ,'server' ,'information' ,'co.th' ,'sg' ,'whois.nic.nf' ,'net.cn' ,'for-our.info' ,'webnode' ,'org.au' ,'wixsite' ,'com.co' ,'design.1' ,'my' ,'whois.domaindiscover.com' ,'review.1' ,'CO' ,'whois.acens.net' ,'zw' ,'sicher' ,'solutions.1' ,'co.zw' ,'recovery' ,'ba.1' ,'p' ,'whois.yesnic.com' ,'whois.cronon.net' ,'whois.nominalia.com' ,'usa' ,'science.1' ,'t' ,'is.1' ,'files' ,'inet' ,'rs' ,'blogspot.com.br' ,'confirm' ,'ec.1' ,'nu.1' ,'whois.rumahweb.com' ,'ipaddress' ,'sch.id' ,'wap' ,'stream.1' ,'sch' ,'forum' ,'SG' ,'f' ,'whois.your-server.de' ,'com.uy' ,'issue' ,'1' ,'bg' ,'lnx' ,'nu' ,'com.hk' ,'check' ,'tudown' ,'science' ,'cr' ,'virus' ,'md.1' ,'eg' ,'whois.pairDomains.com' ,'whois.dinahosting.com' ,'apps' ,'ma.1' ,'https' ,'autodiscover' ,'view' ,'sign' ,'SC' ,'u' ,'adv' ,'file' ,'2016' ,'ind' ,'photo' ,'whois.udag.net' ,'video' ,'net.br' ,'download' ,'sicherheitssystem' ,'team' ,'space' ,'whois.melbourneit.com' ,'atasoyzeminmarket' ,'webmail' ,'inc' ,'jobs' ,'www1' ,'live.1' ,'kr' ,'secured' ,'AR' ,'portal' ,'hr' ,'hmrc' ,'admin' ,'auto' ,'org.in' ,'and' ,'studio' ,'tz' ,'deviantart' ,'cloud' ,'edu.cn' ,'stream' ,'HU' ,'clients' ,'windows' ,'IL' ,'PH' ,'EG' ,'hosting' ,'client' ,'whois.eurodns.com' ,'ne' ,'whois.imena.ua' ,'wix' ,'pe' ,'whois.namefull.com' ,'whois.ibi.net' ,'RO' ,'img' ,'sn' ,'whois.regtons.com' ,'life' ,'bank' ,'pixnet' ,'trade.1' ,'http://api.fastdomain.com/cgi/whois' ,'blogspot.in' ,'ws' ,'la' ,'py' ,'CZ' ,'logon' ,'ma' ,'md' ,'hk' ,'bo' ,'office' ,'lv.1' ,'pdf' ,'lv' ,'whois.zenregistry.com' ,'to.1' ,'ip' ,'bbs' ,'postesecurelogin' ,'composesite' ,'SE' ,'customer' ,'ads' ,'iteye' ,'whois.akamai.com' ,'r' ,'one.1' ,'global' ,'dns' ,'internet' ,'of' ,'eco' ,'network' ,'assets' ,'re.1' ,'whois.nic.la' ,'data' ,'limited' ,'whois.namejuice.com' ,'sex' ,'org.pk' ,'hotels' ,'mk.1' ,'UG' ,'whois.namebright.com' ,'wellsfargo' ,'g' ,'whois.aitdomains.com' ,'uk' ,'webcam.1' ,'AUSTRALIA' ,'whois.star-domain.jp' ,'all' ,'ucoz' ,'manage' ,'confirmation' ,'vip' ,'y' ,'do' ,'world.1' ,'staging' ,'whois.namesecure.com' ,'ssl' ,'today.1' ,'business' ,'host.1' ,'whois.101domain.com' ,'log' ,'objectstorage' ,'up' ,'sh' ,'orange' ,'sport' ,'software' ,'old' ,'pc' ,'mt' ,'sms' ,'whois.do-reg.jp' ,'ac.in' ,'Malaysia' ,'2017' ,'contact' ,'whois.in2net.com' ,'v' ,'name.1' ,'auth' ,'pp.ua' ,'london' ,'dl' ,'whois.pananames.com' ,'who.godaddy.com/' ,'PT' ,'whois.namebay.com' ,'now' ,'gov' ,'city' ,'med' ,'game' ,'acc' ,'doc' ,'whois.netart-registrar.com' ,'asia' ,'SA' ,'track' ,'br' ,'0.1' ,'x' ,'india' ,'pissedconsumer' ,'restaurant' ,'ee.1' ,'GBRAINE' ,'k' ,'user' ,'posta' ,'drive' ,'rssing' ,'computer' ,'ind.br' ,'org.ua' ,'whois.lcn.com' ,'cgi' ,'co.rs' ,'pay' ,'feedmybeta' ,'sourceforge' ,'co.tz' ,'ms' ,'blogfa' ,'best' ,'az.1' ,'com.ph' ,'n' ,'ar.1' ,'am.1' ,'company' ,'name' ,'dynamic' ,'cat.1' ,'consulting' ,'ad' ,'autenticazione' ,'whois.instra.net' ,'today' ,'ac.id' ,'images' ,'com.sa' ,'clan' ,'spb' ,'ps' ,'payment' ,'ao' ,'net.in' ,'notification' ,'ph' ,'co.ao' ,'nkjr45f6' ,'tn.1' ,'docs' ,'l' ,'BE' ,'is' ,'work.1' ,'org.cn' ,'location' ,'travel.1' ,'spc' ,'GR' ,'gob' ,'whois.dyndns.com' ,'action' ,'digital' ,'click' ,'BS' ,'ve.1' ,'foo' ,'tw' ,'org.mx' ,'tools' ,'j' ,'www2' ,'CY' ,'2' ,'CR' ,'player' ,'w' ,'pk.2' ,'netflix' ,'ny23u2' ,'pa' ,'deutschland' ,'events' ,'fang' ,'review' ,'lb' ,'book' ,'webcam' ,'trade' ,'party.1' ,'smart' ,'whois.interdominios.com' ,'management' ,'com.py' ,'TW' ,'DK' ,'whois.inames.co.kr' ,'date.1' ,'whois.tppwholesale.com.au' ,'expert.1' ,'herokuapp' ,'skyrock' ,'ku4346b74bi' ,'search' ,'bancaposta' ,'code' ,'fm.1' ,'seo' ,'play' ,'fashion' ,'com.mt' ,'france' ,'seesaa' ,'html' ,'car' ,'games' ,'house' ,'box' ,'star' ,'int' ,'ns' ,'training' ,'report' ,'edu.co' ,'intl' ,'whois.Namescout.com' ,'NO' ,'open' ,'beauty' ,'validierung' ,'men.1' ,'st' ,'plus' ,'herokuapp.com' ,'srv' ,'to' ,'lu.1' ,'ne.jp' ,'cat' ,'akamaihd' ,'bd' ,'safety' ,'ee' ,'villa' ,'english' ,'rocks' ,'accountant.1' ,'music' ,'newsletter' ,'whois.domainsite.com' ,'marketing' ,'1.1' ,'ww1' ,'refund' ,'call' ,'gt' ,'whois.Rebel.ca' ,'beta' ,'alibaba' ,'myaccount' ,'battle' ,'LB' ,'or.id' ,'rocks.1' ,'international' ,'ww2' ,'solutions' ,'guide' ,'VE' ,'taxi' ,'blogspot.co.uk' ,'social' ,'connect' ,'whois.easydns.com' ,'edu.pk' ,'wp' ,'school' ,'appspot.com' ,'appspot' ,'al' ,'sc' ,'salon' ,'whois.fabulous.com' ,'yolasite.com' ,'billing' ,'sklep' ,'bz.1' ,'ly.1' ,'4' ,'sicherheit' ,'party' ,'pr' ,'products' ,'o' ,'edu.in' ,'watch' ,'spa' ,'or.kr' ,'bin' ,'phone' ,'direct' ,'you' ,'telemetryverification' ,'wiki' ,'yolasite' ,'accountant' ,'dc' ,'blogspot.com.es' ,'locked' ,'gen' ,'BG' ,'green' ,'typepad' ,'vu' ,'FI' ,'benutzer' ,'net.pl' ,'org.nz' ,'tistory' ,'start' ,'red' ,'dz' ,'monsite' ,'image' ,'dr' ,'community' ,'whois.easyspace.com' ,'map' ,'loginlink' ,'AF' ,'db' ,'west' ,'du' ,'rfihub' ,'mi' ,'0' ,'org.tw' ,'lu' ,'find' ,'prime' ,'so' ,'blogspot.ca' ,'law' ,'crm' ,'gast' ,'list' ,'ba' ,'homestead' ,'li.1' ,'z' ,'squeeze549' ,'da' ,'express' ,'warning' ,'power' ,'qa' ,'event' ,'com.ec' ,'center' ,'in.ua' ,'maps' ,'b34in3i' ,'org.ar' ,'profile' ,'blogspot.ru' ,'whois.ovh.net' ,'go.id' ,'pub' ,'updates' ,'technology' ,'ltd' ,'kenntnis' ,'Whois.communigal.net' ,'work' ,'edu.mx' ,'com.es' ,'project' ,'6' ,'whois.enterprice.net' ,'HR' ,'NZ' ,'ww' ,'research' ,'az' ,'tour' ,'videos' ,'cs' ,'solution' ,'care' ,'android' ,'date' ,'ly' ,'cafe' ,'members' ,'webs' ,'projects' ,'org.sg' ,'ftp' ,'eng' ,'sicherheitshilfe' ,'lab' ,'store' ,'lady' ,'job' ,'services' ,'mn.1' ,'am' ,'password' ,'ac.th' ,'official' ,'gallery' ,'bigcartel' ,'informer' ,'racing' ,'as' ,'sports' ,'googlevideo' ,'news' ,'h' ,'myshopify' ,'canalblog' ,'edu.vn' ,'men' ,'org.pl' ,'le' ,'cms' ,'centre' ,'ci.1' ,'tokyo.1' ,'content' ,'js' ,'cricket.1' ,'get' ,'style' ,'europe' ,'germany' ,'coop.1' ,'verbraucher' ,'waw.pl' ,'tw.2' ,'nifty' ,'kunden' ,'gov.in' ,'li' ,'24' ,'sv' ,'careers' ,'bonus' ,'tours' ,'blogspot.de' ,'tn' ,'fitness' ,'found' ,'im.1' ,'mediafire' ,'bz' ,'radio' ,'agency.1' ,'storno' ,'softonic' ,'local' ,'pics.1' ,'hd' ,'7' ,'footprintdns' ,'ec' ,'master' ,'schutz' ,'org.il' ,'ent' ,'personal' ,'faith' ,'or.jp' ,'sd' ,'whois.blacknight.com' ,'rhcloud.com' ,'web.id' ,'deu' ,'buy' ,'mn' ,'medical' ,'tracking' ,'form' ,'barclays' ,'edu.pl' ,'storage' ,'3020' ,'over' ,'yahoo' ,'bar' ,'gen.tr' ,'learn' ,'amber' ,'re' ,'promo' ,'jd' ,'ai.1' ,'love' ,'clo' ,'cm' ,'chat' ,'paris' ,'photos' ,'amazonaws' ,'one' ,'intranet' ,'survey' ,'SK' ,'us.2' ,'vc' ,'res' ,'real' ,'kiev' ,'magazine' ,'soft' ,'analytics' ,'blogspot.co.id' ,'my.2' ,'s3' ,'www3' ,'cocolog' ,'easy' ,'prod' ,'hi' ,'mk' ,'forums' ,'bmw' ,'host' ,'helpdesk' ,'kiev.ua' ,'fm' ,'academy' ,'nachweis' ,'el' ,'RS' ,'org.tr' ,'blogspot.com.tr' ,'tickets' ,'dental' ,'go.th' ,'divorce' ,'azurewebsites.net' ,'ac.kr' ,'angabe' ,'ta' ,'cfapps' ,'ny' ,'clinic' ,'mitteilung' ,'protect' ,'foto' ,'org.za' ,'tc' ,'park' ,'immobilien' ,'panel' ,'sakura' ,'links' ,'whois.comlaude.com' ,'ds' ,'energy' ,'sec' ,'line' ,'time' ,'LU' ,'golf' ,'SI' ,'central' ,'ww38' ,'press.1' ,'tx' ,'5' ,'et' ,'na' ,'east' ,'mo' ,'163' ,'8' ,'xml' ,'bio' ,'sp' ,'air' ,'uk.com' ,'tokyo' ,'mit' ,'stats' ,'rewards' ,'com.pt' ,'archive' ,'live' ,'systems' ,'exblog' ,'ovh.1' ,'blogs' ,'zone' ,'nature' ,'land' ,'blogspot.com.au' ,'und' ,'galerie' ,'im' ,'BB' ,'world' ,'k12' ,'main' ,'ac.uk' ,'MT' ,'s3.amazonaws.com' ,'edu.tw' ,'les' ,'52' ,'market' ,'8.1' ,'bs' ,'italia' ,'media' ,'uptodown' ,'cylex' ,'edu.au' ,'berlin' ,'edu.my' ,'nz' ,'downloads' ,'share' ,'ccterminalcloud' ,'gadget' ,'blogspot.com.ar' ,'corp' ,'pizza' ,'piwik' ,'agency' ,'coop' ,'gmbh' ,'lite' ,'design' ,'px' ,'bau' ,'booking' ,'com.eg' ,'blogspot.mx' ,'lyncdiscover' ,'uni' ,'co.at' ,'education' ,'va' ,'blogspot.jp' ,'nic' ,'blogspot.it' ,'on.ca' ,'ai' ,'quietly' ,'mybigcommerce' ,'parts' ,'in.th' ,'praxis' ,'blogspot.tw' ,'sh.cn' ,'54' ,'10' ,'blogspot.fr' ,'immo' ,'wikispaces' ,'wa' ,'lp' ,'nic.in' ,'ovh' ,'spb.ru' ,'whois.domrobot.com' ,'card' ,'edu.br' ,'VG' ,'cdn2' ,'finam' ,'library' ,'go.kr' ,'2.1' ,'lib' ,'nc' ,'ap' ,'wpengine' ,'me.uk' ,'pics' ,'1024sj' ,'sites' ,'org.hk' ,'email' ,'i2' ,'em' ,'cricket' ,'s1' ,'des' ,'fr.2' ,'travel' ,'der' ,'id.2' ,'hk.2' ,'blogspot.nl' ,'mc' ,'camping' ,'slack' ,'ninja.1' ,'1688' ,'partner' ,'cedexis' ,'gov.tr' ,'cdn1' ,'japan' ,'press' ,'ci' ,'gob.mx' ,'v6exp3' ,'pe.2' ,'diabetes' ,'bandcamp' ,'i1' ,'infusionsoft' ,'vast' ,'hamburg' ,'ac.jp' ,'muenchen' ,'tl' ,'ninja' ,'ct' ,'comune' ,'metric' ,'whois.rebel.ca' ,'radar' ,'ec2' ,'blogspot.gr' ,'blogspot.kr' ,'ed' ,'v4' ,'gstatic' ,'jimdo' ,'gov.ua' ,'es.2' ,'academia' ,'gov.it' ,'pubsub' ,'ag' ,'edu.ar' ,'init' ,'gw' ,'compute' ,'qc.ca' ,'emc' ,'if' ,'netdna' ,'expert' ,'btrll' ,'sg.2' ,'openx' ,'gov.au' ,'_' ,'gr.jp' ,'edu.tr' ,'gov.uk' ,'edu.hk' ,'cafe24' ,'die' ,'ac.il' ,'blogspot.com.eg' ,'r4' ,'kim.1' ,'hateblo' ,'force' ,'fbjs' ,'gov.tw' ,'p5' ,'nhs.uk' ,'Whois.55hl.com' ,'haus' ,'aero' ,'whois.communigal.net' ,'wufoo' ,'mx.2' ,'squarespace' ,'jugem' ,'ed.jp' ,'sharepoint' ,'p4' ,'kim' ,'iheart' ,'frankfurt' ,'manage1' ,'go.jp' ,'feuerwehr' ,'tr.1' ,'ir.2' ,'r2' ,'r1' ,'ca.2' ,'r6' ,'smugmug' ,'ph.2' ,'cr.1' ,'pa.1' ,'gb' ,'ru.2' ,'r5' ,'whois.namescout.com' ,'Ch' ,'ug' ,'nl.2' ,'cz.2' ,'whois.pairdomains.com' ,'at.2' ,'se.2']
Top300TLD=['com.au', 'de', 'cn', 'com.br', 'blogspot.com', 'nl', 'jp', 'ru', 'cl', 'cf', 'co.jp', 'co.uk', 'info', 'ga', 'ml', 'com', 'edu', 'co.id', 'fr', 'blogspot.ie', 'cc', 'online', 'gq', 'tk', 'ch', 'at', 'com.ng', 'ro', 'org', 'id', 'cloudfront.net', 'be', 'io', 'co.il', 'in', 'gdn', 'it', 'club', 'sch.id', 'github.io', 'ie', 'co.kr', 'se', 'com.ve', 'nf', 'xyz', 'gov', 'pp.ua', 'cz', 'blogspot.in', 'com.pk', 'loan', 'for-our.info', 'org.uk', 'top', 'ne.jp', 'eu', 'dk', 'com.ar', 'kz', 'win', 'or.jp', 'pk', 'ir', 'fi', 'tech', 'ac.kr', 'co.za', 'ac.jp', 'co.in', 'la', 'ac.uk', 'us', 'co.zw', 'go.kr', 'com.my', 'com.tw', 'gov.cn', 'no', 'edu.cn', 'ge', 'com.uy', 'blogspot.com.br', 'hk', 'tv', 'com.pe', 'com.mx', 'or.id', 'ua', 'kr', 'com.hk', 'org.pk', 'cat', 'sk', 'vu', 'ac.id', 'co.ao', 'gov.in', 'gov.uk', 'lk', 'men', 'blogspot.co.uk', 'com.vn', 'website', 'appspot.com', 'es', 'ed.jp', 'nic.in', 'pw', 'net', 'net.cn', 'blogspot.mx', 'ca', 'gov.tw', 'pe', 'co.rs', 'vn', 'by', 'blogspot.jp', 'tokyo', 'or.kr', 'org.il', 'ovh', 'br', 'edu.tw', 'pics', 'gov.au', 'go.jp', 'com.sa', 'web.id', 'co.tz', 'net.au', 'lu', 'ee', 'is', 'spb.ru', 'nhs.uk', 'mn', 'co.nz', 'com.py', 'stream', 'net.br', 'ind.br', 'tw', 'webcam', 'blogspot.fr', 'go.th', 'biz', 'site', 'com.ua', 'to', 're', 'edu.tr', 'edu.au', 'su', 'go.id', 'ninja', 'edu.pk', 'gov.tr', 'gob.mx', 'my', 'co.th', 'hu', 'com.cn', 'org.in', 'party', 'pt', 'blogspot.co.id', 'com.co', 'gr.jp', 'uk.com', 'org.br', 'com.mt', 'kim', 'cloud', 'edu.ar', 'ma', 'bid', 'center', 'press', 'blogspot.gr', 'blogspot.it', 'me.uk', 'org.mx', 'qc.ca', 'blogspot.tw', 'ac.il', 'vc', 'lv', 'pro', 'lt', 'ba', 'blogspot.com.ar', 'ac.in', 'pl', 'store', 'solutions', 'blogspot.com.au', 'net.in', 'rs', 'ae', 'nz', 'tl', 'faith', 'mx', 'blogspot.nl', 'al', 'im', 'life', 'md', 'travel', 'nu', 'aero', 'yolasite.com', 'edu.hk', 'accountant', 'az', 'org.au', 'gov.it', 'org.hk', 'on.ca', 'click', 'gov.ua', 'edu.br', 'ai', 'science', 'co', 'host', 'sh.cn', 'gr', 'uk', 'cricket', 'com.eg', 'rhcloud.com', 'asia', 'date', 'download', 'email', 'link', 'co.at', 'com.ec', 'blogspot.com.eg', 'org.ua', 'edu.co', 'blogspot.com.tr', 'edu.vn', 'kiev.ua', 'me', 'news', 'blogspot.kr', 'azurewebsites.net', 'org.nz', 'expert', 'design', 'ci', 'name', 'org.tw', 'coop', 'com.pl', 'rocks', 'com.tr', 'services', 'fm', 'si', 'org.za', 'work', 'mobi', 'blogspot.com.es', 'com.ph', 'com.es', 'blogspot.ca', 'in.th', 'in.ua', 'org.pl', 'edu.my', 'media', 'ly', 'bg', 'edu.mx', 'ws', 'blogspot.de', 'trade', 'review', 'org.tr', 's3.amazonaws.com', 'ac.th', 'herokuapp.com', 'agency', 'blogspot.ru', 'edu.pl', 'net.pl', 'am', 'sg', 'tn', 'hr', 'space', 'mk', 'waw.pl', 'gen.tr', 'today'] 
Top100Server=['com.au', 'au', 'bodydesign', 'www', 'de', 'whois.markmonitor.com', 'de.1', 'whois.networksolutions.com', 'whois.ename.com', 'No_Data.1', 'jp.1', 'cn', 'whois.PublicDomainRegistry.com', 'com.br', 'paypal', 'blogspot', 'br.1', 'JP', 'account', 'US', 'id.1', 'blogspot.com', 'No_Data', 'apple', 'nl', 'cn.1', 'nl.1', 'jp', 'esy', 'whois.discount-domain.com', 'com.1', '000webhostapp', 'appleid', 'login', 'tumblr', 'wordpress', 'whois.hostinger.com', 'info.1', 'signin', 'ru', 'whois.corporatedomains.com', 'DE', 'update', 'ru.1', 'cl', 'cf', 'cl.1', 'co.jp', 'cf.1', 'co.uk', 'sophosxl', 'info', 'CN', 'uk.1', 'support', 'GB', 'ga', 'ml', 'com', 'ml.1', 's', 'ga.1', 'whois.web.com', 'edu', 'blog', 'co.id', 'whois.bizcn.com', 'icloud', 'ie.1', 'fr', 'verify', 'security', 'online.1', 'blogspot.ie', 'abuse', 'pubnub', 'io.1', 'cn.2', 'verification', 'whois.ovh.com', 'myjino', 'ebay', 'whois.jprs.jp', 'whois.psi-usa.info', 'whois.register.com', 'cc', 'fr.1', 'system', 'online', 'gq', 'service', 'gq.1', 'cc.1', 'tk', 'kr.1', 'tk.1', 'whois.1and1.com', 'hol', 'tekblue', 'ID']
Top500Ngram=['d9c', '675', '0a1', 'd5d', '9j5', 'jnq', '1f5', '575', '9cb', 'f54', '7sb', 'd80', 'btd', 'td9', 'c0d', '7jn', 'td5', '07s', 'sbt', 'j57', 'dyd', '57j', 'q9j', '13c', '3c0', 'a13', '757', '5d8', 'cbh', 'db1', 'b1f', 'nq9', '754', '80a', '107', 'bhs', 'ody', '0db', 'hsb', '546', '467', 'ign', 'bod', 'gn.', 'yde', 'sig', '.bo', 'des.1', 'esi', 'om.', '.au', 'm.a', 'n.c', 'pot', 'ww.', 'pay.1', 'blo', 'spo', 'gsp', 'ogs', 'ot.', 'ypa', 'acc.1', 'pal', '.id', '.bl', 'm.b', 'cco', '.de', 'lr.', 'ple', 'ppl', 'mbl', 'blr', 's.n', 'ecu', 'ppo', '.nl', 'log.1', 'hin', '.br', 'sec.1', '.jp', 'cti', '.cn', 'ide', 'umb', '.tu', 'unt', 'l.n', 't.c', '.so', 'w.n', 'ub.', '.ca', 'sha', '-se', 'w.j', 'tum', 'tio', 'rvi', 'ena', 'mas', 'est', 'out', '.th', 'ic.', 'sio', 'deo', 'ebo', 'oun', 'upp', 'vic', 'doc.1', 'ein', 'ber', 'ron', 'e.r', 'dan', 'red.1', '.be', 'ser', '.ch', 'hel', 'ice', 'tch', 'for.1', 'lic', 'nes', 'ten', 'loc', 'ges', 'in.', '.us', 'og.', '.al', 'as.', 'erv', 'ogi', 'id.', 'tru', 'com.2', 'tem', 'shi', 'ach', 'len', '.hu', 'aci', 'rke', 'ars', 'ics', 'sup', 'el.', 'san', 'cur', 'cou', 'w.c', 'ami', 'vid', 'dia', '.ho', 'v.c', 'cit', 'oll', 'lia', 'een', 'lan', 'tte', 'mic', 't.n', 'us.', 'chi', 'use', 'n.d', 'nts', 'w.w', 'thi', 'ay.', 'ure', 'r.c', 'dat', 'hea', 'du.', 'rou', 'ace', 'gin', 'app.1', 'ner', 's.c', 'e.o', '.il', 'hil', 'w.t', 'rte', 'iti', 'gar', 'ima', 'tor', 'avi', 'the.1', 'ida', 'tom', 'tho', '.sh', 'her', 'ker', 'hou', 'ute', '.no', 'che', 'w.r', 'fac', 'mai', 'cor', 'rde', 'isi', 'tan', 'sol', 'ght', 'res.1', 'ws.', 'o.u', 't.d', 'sta', 'en-', 'ene', 'se.', 'nor', 'rec', 'ous', 'sho', 'mus', 'rk.', 'ol.', 'tel', '.uk', '201', 'ina', 'er.', 'lut', 'xyz.2', 'bea', 'lab.1', 'cro', 'w.l', 'evi', 'erc', 'es.', 'can', 'sel', 'roc', '.di', 'cle', '.ga', '.si', 'nic.1', 'ote', 'w.f', 'ran', '.xy', 'ts.', 'ish', 't.i', '.la', 's.b', 'eca', 'ana', 'ble', 'll.', 'ine', 'ali', 'ope', 'nne', 'pp.', 'ons', '.me', 'eat', 'eve', 'ede', 'o.j', 'por', 'ade', 'bur', 'dow', 'ari', 'ino', 'sit', 'bri', 'ger', '.pr', 'ne.', 'olu', 'hot', 'w.s', 'ion', 'tim', 'w.o', 'sur', 'ans', 'rs.', 'aro', '.go', 'rus', 'd.c', 'oup', 'ria', 'ove', 'erg', 's.s', 'min', '.sp', '.wo', 'cdn.1', 'ica', 'ken', 'cre', 'emi', 'att', '.co', 'you.1', 'end', 'usi', '.do', 'igh', 'dev.1', 'gam', 'mar', '-ma', 'gro', 'ale', 'rav', 'ape', 'tri', 'ada', 'e.d', '.fo', 'bil', 'eba', 'ont', 'sin', 'ult', 'ill', 'lth', '.le', 'sso', '.ha', '.wi', 'ema', 'tis', 'omm', 'alt', 'ter', 'ame', 'co.', 'ed.', 'inc.1', 'ct.', 'k.c', 'adi', 'rep', 'bel', 'ega', 'ost', 'int.1', 'ien', 'pag', 'dio', 'han', 'vie', 'nga', 'ban', 'api.1', 'nas', 'rth', '-re', '.ru', 'ds.', 'fir', 'ngs', 'orl', 'and.1', 'ins', 'let', 'urs', 'sys', 'ens', 'ama', 'lou', 'tha', 'ise', '.fr', 'nal', 'up.', 'spi', 'omp', 'uti', 'm.m', 'rit', 'rne', 'rse', 'emo', 'ash', 'pub.1', 'anc', 'enc', 'c.c', 'arr', 'e.b', 'uri', 'ps.', 'le-', 'sma', 'mal', 'rie', 'ns.', '.in', 'ket', 'ead', 'cto', 'qui', 'esa', 'ess', '.ac', 'oba', 'udi', 'rai', 'nt.', 'cap', 'ond', 'ta.', 'nsu', 'o.i', 'ann', 'dri', 'rna', 'rom', 'ry.', 'edi', 'owe', 'i.c', 'tas', 'nfo', 'rag', 'met', 'dem', 'lec', 'hat', 'er-', 'asa', 'n.o', 'ret', 'win.2', 'nin', 'hub.1', 'it.', 'tur', 'ona', 'or.', '.sa', 'riv', 'ort', 'go.', 'al.', 'en.', 'sic', 'fil', 'cia', 'w.b', 'fre', 'act', 'w.u', '.gr', 'ain', 'bar.1', 'get.1', 'hon', 'onl', 'le.', 'sou', 'tiv', 'eas', 'eng.1', 'ele', 'eco.1', 'x.c', 'sto', 'ase', 'gra', '.ma', 'yst', 'rch', 'tic', 'ls.', 'rel', 'off', 'ado', 'gol', '.ie', 'rop', 'ood'] 
Top200BOW=['au', 'bodydesign', 'www', 'de', 'jp', 'blogspot', 'paypal', 'br', 'account', 'id', 'apple', 'cn', 'nl', 'esy', 'com', '000webhostapp', 'appleid', 'login', 'tumblr', 'wordpress', 'info', 'signin', 'update', 'ru', 'cf', 'cl', 'sophosxl', 'support', 'uk', 'ga', 'ml', 's', 'blog', 'ie', 'icloud', 'online', 'verify', 'security', 'pubnub', 'verification', 'io', 'myjino', 'fr', 'ebay', 'system', 'service', 'gq', 'cc', 'kr', 'hol', 'tk', 'tekblue', 'secure', 'co', 'googlevideo', 'il', 'ng', 'ch', 'cloudfront', 'org', 'alert', 'pk', 'be', 'microsoft', 'cdn', 'ro', 'information', 'fc2', 'confirm', 'webapps', 'gdn', 'cgi', 'hatenablog', 'fb', 'hub', 'our', 'sn', '3', 'bt', 'mdmag', 'sandwichdrip', 'amazon', 'pagesstudy', 'page', 'weebly', 'xn', 'pe', 'it', 'posta', 'github', 'se', 've', 'in', 'bankofamerica', 'disqus', 'pixnet', 'xyz', 'issue', 'ipaddress', 'accounts', 'foo', '16mb', 'refund', 'nf', 'cz', 'facebook', 'for', '0.1', 'club', 'confirmation', 'wikia', 'html', 'error', 'tw', 'cutestat', 'fang', 'tistory', 'usaa', 'https', 'objectstorage', 'metric', 'virus', 'pages', 'static', 'slack', 'v6exp3', 'sign', 'inet', 'gast', 'recovery', 'readthedocs', 'hotel', 'loan', 'limited', 'wellsfargo', 'chase', 'sicher', 'hmrc', 'dropbox', 'netflix', 'beget', 'access', 'postesecurelogin', 'abqrm', 'sicherheit', '1.1', 'gstatic', 'us', '5gbfree', 'logon', 'i', 'secured', 'dk', 'hk', 'kz', 'shop', 'eu', 'deviantart', 'xml', 'help', 'api', 'sicherheitssystem', 'url_', 'schutz', '96', 'rssing', 'at', 'amber', 'warning', 'center', 'softonic', 'deu', 'divorce', 'barclays', 'safety', 'ne', 'top', 'fi', 'atasoyzeminmarket', 'locked', 'found', 'customer', 'iteye', 'cocolog', 'ku4346b74bi', 'b34in3i', 'ny23u2', 'nkjr45f6', 'sitey', 'call', '1.2', 'win', 'auth', 'payment', 'your', 'telemetryverification', 'bancaposta', 'blogfa', '8', 'jimdo']
Top500=['com.au', 'au', 'bodydesign', 'www', 'de', 'whois.markmonitor.com', 'de.1', 'whois.networksolutions.com', 'whois.ename.com', 'No_Data.1', 'jp.1', 'cn', 'whois.PublicDomainRegistry.com', 'com.br', 'paypal', 'blogspot', 'br.1', 'JP', 'account', 'US', 'id.1', 'blogspot.com', 'No_Data', 'apple', 'nl', 'cn.1', 'nl.1', 'jp', 'esy', 'whois.discount-domain.com', 'com.1', '000webhostapp', 'appleid', 'login', 'tumblr', 'wordpress', 'whois.hostinger.com', 'info.1', 'signin', 'ru', 'whois.corporatedomains.com', 'DE', 'update', 'ru.1', 'cl', 'cf', 'cl.1', 'co.jp', 'cf.1', 'co.uk', 'info', 'sophosxl', 'CN', 'uk.1', 'support', 'ga', 'GB', 'ml', 'com', 'ml.1', 's', 'ga.1', 'whois.web.com', 'edu', 'blog', 'co.id', 'whois.bizcn.com', 'icloud', 'ie.1', 'fr', 'verify', 'security', 'online.1', 'blogspot.ie', 'abuse', 'pubnub', 'io.1', 'cn.2', 'verification', 'whois.ovh.com', 'myjino', 'ebay', 'whois.jprs.jp', 'whois.psi-usa.info', 'whois.register.com', 'cc', 'fr.1', 'system', 'online', 'gq', 'service', 'gq.1', 'cc.1', 'tk', 'kr.1', 'tk.1', 'whois.1and1.com', 'hol', 'co.1', 'tekblue', 'ID', 'ch', 'whois.gandi.net', 'at', 'com.ng', 'PA', 'ng', 'googlevideo', 'il', 'secure', 'ro', 'org', 'whois.lexsynergy.com', 'whois.paycenter.com.cn', 'id', 'microsoft', 'org.1', 'cloudfront.net', 'pk.1', 'cloudfront', 'alert', 'ch.1', 'be', 'io', 'ro.1', 'co.il', 'cdn', 'be.1', 'whois.tucows.com', 'in', 'KR', 'webapps', 'information', 'whois.safenames.net', 'fc2', 'whois.meshdigital.com', 'gdn.1', 'gdn', 'whois.synergywholesale.com', 'whois.ilovewww.com', 'whois.ascio.com', 'it', 'confirm', 'our', 'fb', 'hub', '3', 'hatenablog', 'sn', 'cgi', 'club', 'amazon', 'sandwichdrip', 'mdmag', 'page', 'bt', 'pagesstudy', 'pe.1', 'ES', 'whois.maff.com', 'whois.NameBright.com', 'weebly', 'github', 'whois.godaddy.com', 'posta', 'xn', 've', 'sch.id', 'github.io', 'whois.evonames.com', 'bankofamerica', 'it.1', 'se.1', 'ie', 'pixnet', 'co.kr', 'in.1', 'disqus', 'foo', 'whois.namecheap.com', 'IN', 'se', 'com.ve', 'xyz.1', 'nf', 'xyz', 'issue', 'ipaddress', 'whois.nic.nf', 'gov', 'facebook', '16mb', 'for', 'accounts', 'cz.1', 'pp.ua', 'cz', '0.1', 'club.1', 'whois.eranet.com', 'nf.1', 'refund', 'blogspot.in', 'html', 'wikia', 'confirmation', 'com.pk', 'usaa', 'error', 'loan', 'cutestat', 'whois.rrpproxy.net', 'whois.net-chinese.com.tw', 'fang', 'https', 'inet', 'recovery', 'objectstorage', 'NL', 'gast', 'hotel', 'loan.1', 'whois.ibi.net', 'pages', 'for-our.info', 'whois.launchpad.com', 'org.uk', 'metric', 'static', 'virus', 'sicherheit', 'readthedocs', 'chase', 'tistory', 'tw.1', 'netflix', 'v6exp3', 'slack', 'sign', 'dropbox', 'top', 'hk.1', 'postesecurelogin', 'i', 'shop', 'beget', 'hmrc', 'logon', 'grs-whois.cndns.com', 'eu.1', 'secured', 'ne.jp', 'sicher', 'eu', 'wellsfargo', 'd9c', '675', 'abqrm', '5gbfree', '0a1', 'd5d', 'jnq', '9j5', '1f5', '575', 'whois.resellercamp.com', '9cb', 'f54', '7sb', 'd80', 'us.1', 'btd', 'td9', 'c0d', 'td5', '7jn', '07s', 'sbt', 'dyd', '57j', 'j57', 'q9j', '13c', '3c0', 'a13', '757', '5d8', 'cbh', 'url_', 'limited', 'db1', 'dk', 'b1f', 'deu', 'access', 'nq9', '754', '80a', '107', 'bhs', 'ody', '0db', 'hsb', '546', 'dk.1', 'BR', 'whois.star-domain.jp', '1.1', 'com.ar', '467', 'sicherheitssystem', 'deviantart', 'rssing', 'kz', 'kz.1', 'ign', 'bod', 'whois.domaindiscover.com', 'xml', 'warning', 'gstatic', 'api', 'schutz', 'barclays', 'gn.', 'yde', 'center.1', 'help', 'win.1', 'whois.directnic.com', 'win', 'top.1', 'amber', 'or.jp', 'whois.cronon.net', 'sig', 'divorce', 'pk', 'at.1', 'softonic', 'your', 'whois.publicdomainregistry.com', 'ir', 'fi.1', 'sitey', 'ne', 'cocolog', 'locked', 'forum', 'iteye', 'fi', 'customer', 'whois.comlaude.com', '.bo', 'tech', 'found', 'atasoyzeminmarket', 'telemetryverification', 'safety', 'jimdo', 'whois.gabia.com', '96', 'des.1', 'ac.kr', 'ny23u2', 'b34in3i', 'ku4346b74bi', 'nkjr45f6', 'autenticazione', 'bancaposta', 'auth', 'livejournal', 'PE', 'call', 'whois.namefull.com', 'check', '8', 'sicherheitshilfe', 'za', '1.2', 'p', 'co.za', 'autodiscover', 'clan', 'm', 'blogfa', 'whois.pananames.com', 'esi', 'om.', 'NG', 'notification', 'whois.akamai.com', 'ac.jp', 'downyouxi', 'co.in', 'rfihub', 'p5', 'i2', 'lady', 'myaccount', 'bin', 'exblog', 'form', 'whois.internet.bs', 'whois.registrygate.com', 'payment', '.au', 'whois.registrar.amazon.com', 'infusionsoft', 'password', 'pubsub', 'lnx', 'm.a', 'la', 'nifty', 'whois.uniregistrar.net', 'composesite', 'whois.netregistry.com.au', 'TR', 'tv.1', 'MX', 'validierung', 'whois.your-server.de', 'site1', 'sch', 'us', 'ac.uk', 'squeeze549', 'BD', 'en', 'acc', 'protect', 'seesaa', 'i1', 'whois.registrar.eu', 'whois.udag.net', 'whois.55hl.com', 'ar', 'ir.1', 'whois.nicline.com', 'whois.nic.la', 'netdna', 'co.zw', 'akamaihd', 'zw', 'whois.cdmon.com', 'prime', 'tudown', 'nimp', 'com.my', 'go.kr', 'sharepoint', 'bandcamp', 'skyrock', 'incredible', 'ac', 'AT', 'nachweis', 'benutzer', 'verbraucher', 'vn.1', 'down', 'whois.wildwestdomains.com', 'img', 'clo', 'feedmybeta', 'com.tw', 'IL', 'ww2', 'storno', 'intend', 'footprintdns', 'media.1', 'over', 'gov.cn', '51qqxx', 'kenntnis', 'quietly', 'whois.joker.com', 'angabe', 'uy', 'sourceforge', 'whois.onlinenic.com', 'IE', 'no', 'mitteilung', 'hamburg', 'spc', 'edu.cn', 'whois.domain.com', 'whois.blacknight.com', 'go', 'ca.1', '3eeweb', 'twomini', 'ge', 'user', 'whois.west.cn', 'f', 'action']


# ### Run The Next 3 paragrpahs For Subsets of the Data

# In[ ]:


RSA_SUBSET1=RSA.iloc[:,0:1196]
RSA_SUBSET2=RSA.iloc[:,2194:]
#RSA_SUBSET3=RSA[Top200BOW].iloc[:,0:50]
#RSA_SUBSET4=RSA.iloc[:,2299:]
RSA_SUBSET_COMBINED=pd.concat([RSA_SUBSET1,RSA_SUBSET2],axis=1)

# 0 is Malicious
#[1:24] are basic columns
#[24:339] is TLD
#[339:1196] BOW
#[1196:2194] is Ngram
#[2194:2299] is Whois Country
#[2299:] Is Whois Server



# In[ ]:


RSA_FI_SUBSET1=RSA['Malicious']
RSA_FI_SUBSET2=RSA[TOP_FI].iloc[:,0:100]
RSA_FI_SUBSET_COMBINED=pd.concat([RSA_FI_SUBSET1,RSA_FI_SUBSET2],axis=1)


# ## Split Data to test and training

# In[ ]:


X = RSA_SUBSET_COMBINED.drop(['Malicious'],axis=1)
y=RSA_SUBSET_COMBINED['Malicious']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1000)


# ## Run Logistic Regression 

# In[ ]:


logmodel=LogisticRegression()


# In[ ]:


logmodel.fit(X_train,y_train)


# In[ ]:


predictions_LR=logmodel.predict(X_test)


# In[ ]:


print(classification_report(y_test,predictions_LR))


# In[ ]:


# Calculate for every cutoff (check in 0.01 resolution) F1 Score
predictions_LR_prob=logmodel.predict_proba(X_test)
prob_list_LR= [x[1] for x in predictions_LR_prob]

d_LR={}
for threshold in np.arange(0.0, 1.0, 0.01):
    list_for_check_LR=np.int_([y>=threshold for y in prob_list_LR])
    d_LR[threshold]=f1_score(y_test,list_for_check_LR)
df_LR=pd.DataFrame.from_dict(d_LR,orient='index')


# In[ ]:


df_LR[df_LR[0]==df_LR[0].max()]


# In[ ]:


#best cutoff in terms of f1
max_t=0.31
#set predictions according to cutoff
list_for_check_LR_max=np.int_([y>=max_t for y in prob_list_LR])

LR_F1_Score=df_LR[df_LR[0]==df_LR[0].max()]
LR_Recall=recall_score(y_test,list_for_check_LR_max)
LR_Precision=precision_score(y_test,list_for_check_LR_max)

[LR_F1_Score,LR_Recall,LR_Precision]


# ## Run Decision Tree

# In[ ]:


dtree=DecisionTreeClassifier()


# In[ ]:


dtree.fit(X_train,y_train)


# In[ ]:


predictions_DT=dtree.predict(X_test)


# In[ ]:


print(classification_report(y_test,predictions_DT))


# In[ ]:


# Calculate for every cutoff (check in 0.01 resolution) F1 Score
predictions_DT_prob=dtree.predict_proba(X_test)
prob_list_DT= [x[1] for x in predictions_DT_prob]

d_DT={}
for threshold in np.arange(0.0, 1.0, 0.01):
    list_for_check_DT=np.int_([y>=threshold for y in prob_list_DT])
    d_DT[threshold]=f1_score(y_test,list_for_check_DT)
df_DT=pd.DataFrame.from_dict(d_DT,orient='index')


# In[ ]:


df_DT[df_DT[0]==df_DT[0].max()]


# In[ ]:


#best cutoff in terms of f1
max_t=0.35
#set predictions according to cutoff
list_for_check_DT_max=np.int_([y>=max_t for y in prob_list_DT])

DT_F1_Score=df_DT[df_DT[0]==df_DT[0].max()]
DT_Recall=recall_score(y_test,list_for_check_DT_max)
DT_Precision=precision_score(y_test,list_for_check_DT_max)

[DT_F1_Score,DT_Recall,DT_Precision]


# ## Run RandomForest

# In[ ]:


rfc= RandomForestClassifier(n_estimators=500)


# In[ ]:


rfc.fit(X_train,y_train)


# In[ ]:


predictions_RFC=rfc.predict(X_test)


# In[ ]:


print(classification_report(y_test,predictions_RFC))


# In[ ]:


predictions_rfc_prob=rfc.predict_proba(X_test)
prob_list_rfc= [x[1] for x in predictions_rfc_prob]
d_rfc={}
for threshold in np.arange(0.0, 1.0, 0.01):
    list_for_check_rfc=np.int_([y>=threshold for y in prob_list_rfc])
    d_rfc[threshold]=f1_score(y_test,list_for_check_rfc)
df_rfc=pd.DataFrame.from_dict(d_rfc,orient='index')


# In[ ]:


df_rfc[df_rfc[0]==df_rfc[0].max()]


# In[ ]:


#best cutoff in terms of f1
max_t=0.35
#set predictions according to cutoff
list_for_check_rfc_max=np.int_([y>=max_t for y in prob_list_rfc])

rfc_F1_Score=df_rfc[df_rfc[0]==df_rfc[0].max()]
rfc_Recall=recall_score(y_test,list_for_check_rfc_max)
rfc_Precision=precision_score(y_test,list_for_check_rfc_max)

[rfc_F1_Score,rfc_Recall,rfc_Precision]


# #### Random Forest - Features Importance

# In[ ]:


rfc_fi=rfc.feature_importances_

rfc_fi_df=pd.DataFrame(rfc_fi)


# In[ ]:


df_columns=pd.DataFrame(X_train.columns)
df_columns


# In[ ]:


Columns_important = pd.concat([df_columns,rfc_fi_df],axis=1)
Columns_important.columns=['Columns','features_importance']


# In[ ]:


pd.options.display.float_format = '{:,.8f}'.format
Columns_important.sort_values(by=['features_importance'],ascending=False,inplace=True)
Columns_important.to_csv("Feature_Importance.csv")


# In[ ]:


Feature_Important=Columns_important['Columns']
Feature_Important


# #### Random Forest Recall-percision Graph

# In[ ]:


precision, recall, _ = precision_recall_curve(y_test, prob_list_rfc)
average_precision = average_precision_score(y_test, prob_list_rfc)


# In[ ]:


precision, recall, _ = precision_recall_curve(y_test, prob_list_rfc)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))


# ## Run Gradient Boosting

# In[ ]:


GBC= GradientBoostingClassifier()


# In[ ]:


GBC.fit(X_train,y_train)


# In[ ]:


predictions_GBC=GBC.predict(X_test)


# In[ ]:


print(classification_report(y_test,predictions_GBC))


# In[ ]:


predictions_GBC_prob=GBC.predict_proba(X_test)
prob_list_GBC= [x[1] for x in predictions_GBC_prob]
d_GBC={}
for threshold in np.arange(0.0, 1.0, 0.01):
    list_for_check_GBC=np.int_([y>=threshold for y in prob_list_GBC])
    d_GBC[threshold]=f1_score(y_test,list_for_check_GBC)
df_GBC=pd.DataFrame.from_dict(d_GBC,orient='index')


# In[ ]:


df_GBC[df_GBC[0]==df_GBC[0].max()]


# In[ ]:


#best cutoff in terms of f1
max_t=0.31
#set predictions according to cutoff
list_for_check_GBC_max=np.int_([y>=max_t for y in prob_list_GBC])

GBC_F1_Score=df_GBC[df_GBC[0]==df_GBC[0].max()]
GBC_Recall=recall_score(y_test,list_for_check_GBC_max)
GBC_Precision=precision_score(y_test,list_for_check_GBC_max)

[GBC_F1_Score,GBC_Recall,GBC_Precision]

