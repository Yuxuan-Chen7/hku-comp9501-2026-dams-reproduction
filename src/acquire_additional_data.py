"""Retrieve historical public research data, keeping exact provenance and hashes.

Wayback snapshots below were discovered through its CDX index of the original
ECML/PKDD 2006 website, not a differently collected substitute spam corpus.
"""
import hashlib
import argparse
import json
import tarfile
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from .data import ROOT

EMAIL_ARCHIVES = {
    'data_task_a.zip': ('20060429084857', 'http://www.ecmlpkdd2006.org:80/data_task_a.zip'),
    'evaluation_data_labeled.zip': ('20060617074214', 'http://www.ecmlpkdd2006.org:80/evaluation_data_labeled.zip'),
}
EMAIL_SHA256 = {
    'data_task_a.zip': '916e4d07ba526391bab3a8d2e873cd8e3ebae838203c53d3626bf67e2e96cdfc',
    'evaluation_data_labeled.zip': 'dda675d372a516e3fd1daeb893a41b9542c904593dc02c87a9d18ae7da49dd7f',
}


def fetch(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    partial = destination.with_suffix(destination.suffix+'.part')
    for attempt in range(3):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'Academic-dataset-reproduction/1.0'})
            with urllib.request.urlopen(req,timeout=60) as response, partial.open('wb') as output:
                while chunk:=response.read(1024*1024):
                    output.write(chunk)
            if destination.suffix=='.zip' and not zipfile.is_zipfile(partial):
                raise ValueError('Response is not a ZIP archive')
            partial.replace(destination)
            return
        except Exception as exc:
            print('attempt',attempt+1,str(exc),flush=True)
            if attempt==2: raise
            time.sleep(2)


def acquire_email():
    root=ROOT/'data/email_spam/raw'
    records=[]
    for name,(stamp,original) in EMAIL_ARCHIVES.items():
        url=f'https://web.archive.org/web/{stamp}id_/{original}'
        path=root/name
        print('Downloading',url,flush=True)
        fetch(url,path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != EMAIL_SHA256[name]:
            raise ValueError(f'Archive SHA-256 mismatch: {name}')
        with zipfile.ZipFile(path) as archive:
            bad=archive.testzip()
            if bad: raise ValueError(f'ZIP CRC failure: {bad}')
            members=[{'name':m.filename,'bytes':m.file_size,'compressed_bytes':m.compress_size} for m in archive.infolist()]
        record={'file':name,'original_url':original,'retrieval_url':url,
                'snapshot_timestamp':stamp,'bytes':path.stat().st_size,
                'expanded_bytes':sum(m['bytes'] for m in members),
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'members':members,'checked_utc':datetime.now(timezone.utc).isoformat()}
        records.append(record)
        (root/'provenance.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        print(json.dumps(record,indent=2),flush=True)


def acquire_matlab():
    root=ROOT/'data/20ng_matlab/raw'
    path=root/'20news-bydate-matlab.tgz'
    url='http://qwone.com/~jason/20Newsgroups/20news-bydate-matlab.tgz'
    fetch(url,path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    # Hash of the release actually inspected for this reproduction, not an official checksum.
    assert digest=='086fdc442ae2c269c53c6383f641ff92d55cb2b77da19d2cf3562473fc73dd19'
    with tarfile.open(path,'r:gz') as archive:
        members=[{'name':m.name,'bytes':m.size} for m in archive.getmembers()]
    record={'url':url,'bytes':path.stat().st_size,'sha256':digest,'members':members,
            'hash_source':'Locally recorded release hash; publisher checksum unavailable.'}
    (root/'provenance.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    print('Original 20NG MATLAB release verified:',path.stat().st_size,'bytes',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--datasets',nargs='+',choices=['email','matlab'],default=['email','matlab'])
    args=parser.parse_args()
    if 'email' in args.datasets: acquire_email()
    if 'matlab' in args.datasets: acquire_matlab()


if __name__=='__main__': main()
