# -*- coding: utf-8 -*-
import scrapy


class SztukaarchitekturySpider(scrapy.Spider):
    name = 'sztukaarchitektury'
    allowed_domains = ['sztuka-architektury.pl']
    start_urls = [
    	'https://sztuka-architektury.pl/page/14/news-kraj',
    	'https://sztuka-architektury.pl/page/20/news-swiat',
    ]

    custom_settings = {
        'DEPTH_LIMIT': 0
    }

    ARTICLES_URLS = "//section[contains(@class, 'articles-list')]/a/@href"
    NEXT_PAGE_URL = "//nav[contains(@class, 'pagination')]/ul/li/a[contains(@rel, 'next')]/@href"

    TITLE_NODE = "//h1/strong/text()"
    LEAD_NODE = "//div/section[contains(@class, 'text--1')]/p/strong/text()"
    TEXT_NODE = "//div[contains(@class, 'col-md-7')]/section[contains(@class, 'text--1')]/p[not(a)]"

    def parse(self, response):
        # iterate through links to articles
        for quote in response.xpath(self.ARTICLES_URLS):
            yield response.follow(quote, callback=self.parse_article)
            
        #iterate through next page links
        for next_page in response.xpath(self.NEXT_PAGE_URL):
            yield response.follow(next_page, self.parse)
            
    def parse_article(self, response):
        yield {
            'url': response.url,
            'title': response.xpath(self.TITLE_NODE).extract(),
            'lead': response.xpath(self.LEAD_NODE).extract()[:1],
            'text': response.xpath(self.TEXT_NODE).extract()[1:-1],
        }